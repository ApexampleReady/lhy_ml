# Preparing Data

import os
import random
import pandas as pd
import torch
from tqdm import tqdm


# 以下几个函数都是用于concat_feat拼接的
def load_feat(path):
    feat = torch.load(path)
    return feat


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


# 一个phoneme 不会只有一个frame（帧）  训练时接上前后的frame会得到较好的结果
# 这里前后接对称数量，例如concat_n = 11 则前后都接5
def concat_feat(x, concat_n):
    assert concat_n % 2 == 1  # n #为奇数
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(
        1, 0, 2
    )  # concat_n, seq_len, feature_dim
    mid = concat_n // 2
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


# x = torch.tensor([[ 1,  2,  3],
#         [ 4,  5,  6],
#         [ 7,  8,  9],
#         [10, 11, 12]])
# y = concat_feat(x , 3)
# print(y)


def preprocess_data(
    split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337
):
    class_num = 41  # NOTE: pre-computed, should not need change
    mode = "train" if (split == "train" or split == "val") else "test"

    label_dict = {}
    if mode != "test":
        phone_file = open(os.path.join(phone_path, f"{mode}_labels.txt")).readlines()
        # print(os.path.join(phone_path, f'{mode}_labels.txt'))
        for line in phone_file:
            line = line.strip("\n").split(" ")
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == "train" or split == "val":
        # split training and validation data
        usage_list = open(os.path.join(phone_path, "train_split.txt")).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == "train" else usage_list[percent:]
    elif split == "test":
        usage_list = open(os.path.join(phone_path, "test_split.txt")).readlines()
    else:
        raise ValueError("Invalid 'split' argument for dataset: PhoneDataset!")
    # 得到每一个音频代号
    usage_list = [line.strip("\n") for line in usage_list]
    print(
        "[Dataset] - # phone classes: "
        + str(class_num)
        + ", number of utterances for "
        + split
        + ": "
        + str(len(usage_list))
    )

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != "test":
        y = torch.empty(max_len, dtype=torch.long)
    # 将音频数据读取出来 X为特征 y为label
    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f"{fname}.pt"))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != "test":
            label = torch.LongTensor(label_dict[fname])

        X[idx : idx + cur_len, :] = feat
        if mode != "test":
            y[idx : idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != "test":
        y = y[:idx]

    print(f"[INFO] {split} set")
    print(X.shape)
    if mode != "test":
        print(y.shape)
        return X, y
    else:
        return X


# Define Dataset

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


# Define Model

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim, eps=1e-05, momentum=0.1, affine=True),
            #             num_features： 来自期望输入的特征数，C from an expected input of size (N,C,L) or L from input of size (N,L)
            #             eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
            #             momentum： 动态均值和动态方差所使用的动量。默认为0.1。
            #             affine： 一个布尔值，当设为true，给该层添加可学习的仿射变换参数。
            nn.Dropout(0.35),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# 超参数
## Hyper-parameters

# data prarameters
concat_nframes = (
    19  # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
)
train_ratio = (
    0.95  # the ratio of data used for training, the rest will be used for validation
)
#     百万级数据集的训练集验证集划分
#     一种常见的启发式策略是将整体30%的数据用作测试集,这适用于总体数据量规模一般的情况
#    （比如100至10,000个样本）。但在大数据时期，分配比例会发生变化，
#      如100万数据时，98%(训练)1%（验证)1%（测试），超百万时，95%（训练)/2.5%（验证)2.5%（测试)
# -《Machine Learning Yearning》 Andrew Ng

# training parameters
seed = 0  # random seed
batch_size = 1024  # batch size （512）
num_epoch = 100  # the number of training epoch
learning_rate = 0.0001  # learning rate
model_path = "./model.ckpt"  # the path where the checkpoint will be saved

# model parameters
input_dim = (
    39 * concat_nframes
)  # the input dim of the model, you should not change the value
hidden_layers = 2  # the number of hidden layers
hidden_dim = 1024  # the hidden dim


# 对垃圾进行回收所需调用的函数
## Prepare dataset and model
import gc

# preprocess data
train_X, train_y = preprocess_data(
    split="train",
    feat_dir="./libriphone/feat",
    phone_path="./libriphone",
    concat_nframes=concat_nframes,
    train_ratio=train_ratio,
)
val_X, val_y = preprocess_data(
    split="val",
    feat_dir="./libriphone/feat/",
    phone_path="./libriphone",
    concat_nframes=concat_nframes,
    train_ratio=train_ratio,
)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")

import numpy as np


# fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# fix random seed
same_seeds(seed)

# create model, define a loss function, and optimizer
model = Classifier(
    input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=8, T_mult=2, eta_min=learning_rate / 2
)

# #
import torchsummary

torchsummary.summary(model, input_size=(input_dim,))
# TorchSummary提供了更详细的信息分析，包括模块信息（每一层的类型、输出shape和参数量）
# 、模型整体的参数量、模型大小、一次前向或者反向传播需要的内存大小等。
# ncol 设置输出宽度


## Training
best_acc = 0.0
early_stop_count = 0
early_stopping = 8
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    pbar = tqdm(train_loader, ncols=110)  # 用于可视化进度
    pbar.set_description(f"T: {epoch + 1}/{num_epoch}")
    samples = 0
    for i, batch in enumerate(pbar):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        # optimizer.zero_grad()
        # 函数会遍历模型的所有参数，，清空上一次的梯度记录。
        loss = criterion(outputs, labels)  # 设定判别损失函数
        loss.backward()  # 执行反向传播，更新梯度
        optimizer.step()  # 执行参数更新
        # 关于上述函数的讲解 https://blog.csdn.net/PanYHHH/article/details/107361827?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166523672216782391838079%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166523672216782391838079&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-107361827-null-null.142^v52^control,201^v3^add_ask&utm_term=optimizer.step%28%29&spm=1018.2226.3001.4187

        _, train_pred = torch.max(
            outputs, 1
        )  # get the index of the class with the highest probability
        correct = (train_pred.detach() == labels.detach()).sum().item()
        # t.item()将Tensor变量转换为python标量（int float等），其中t是一个Tensor变量，只能是标量，转换后dtype与Tensor的dtype一致
        # detach 该参数的requires_grad 属性设置为False,这样之后的反向传播时就不会更新它

        train_acc += correct
        samples += labels.size(0)
        train_loss += loss.item()
        lr = optimizer.param_groups[0]["lr"]
        # 可视化进度条的参数设置

        pbar.set_postfix(
            {
                "lr": lr,
                "batch acc": correct / labels.size(0),
                "acc": train_acc / samples,
                "loss": train_loss / (i + 1),
            }
        )
    scheduler.step()  # 用于更新学习率
    # 各个情况下的 .step() 一般都是用来更新参数的

    pbar.close()  # 清空并关闭进度条 （progress bar）

    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode#用于将模型变为评估模式，而不是训练模式，这样batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。

        with torch.no_grad():
            pbar = tqdm(val_loader, ncols=110)
            pbar.set_description(f"V: {epoch + 1}/{num_epoch}")
            samples = 0
            for i, batch in enumerate(pbar):
                features, labels = batch  # 取出一个batch中的特征和标签
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)  # 得到预测结果

                loss = criterion(outputs, labels)

                _, val_pred = torch.max(
                    outputs, 1
                )  # get the index of the class with the highest probability
                # 用于得到预测结果
                # torch.max(input: tensor, dim: index)
                # 该函数有两个输入：inputs: tensor，第一个参数为一个张量
                # dim: index，第二个参数为一个整数[-2 - 1]，dim = 0表示计算每列的最大值，dim = 1表示每行的最大值

                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                samples += labels.size(0)
                val_loss += loss.item()
                pbar.set_postfix(
                    {"val acc": val_acc / samples, "val loss": val_loss / (i + 1)}
                )
            pbar.close()
            # 如果模型有进步（在训练集上）就保存一个checkpoint，把模型保存下来

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print("saving model with acc {:.3f}".format(best_acc / len(val_set)))
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stopping:
                # print(f'')中的f使得其有print(''.format())的作用
                print(f"Epoch: {epoch + 1}, model not improving, early stopping.")
                break
    else:
        print("i dont know")
        # print(f'[{epoch + 1:03d}/{num_epoch:03d}] Acc: {acc:3.6f} Loss: {loss:3.6f}')
# print(f'[{epoch + 1:03d}/{num_epoch:03d}] Acc: {acc:3.6f} Loss: {loss:3.6f}')

# 如果没有测试，保存最后一次训练 我们是有测试集的，所以下述代码用不着
# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print("saving model at last epoch")
# 老规矩,清除内存
del train_loader, val_loader
gc.collect()


## Testing## 创造一个测试集用来得到题目想要的预测结果，我们从之前保存的checkpoint也就是最好的模型来预测结果
## Testing
# Create a testing dataset, and load model from the saved checkpoint.

# Create a testing dataset, and load model from the saved checkpoint.

# load data
test_X = preprocess_data(
    split="test",
    feat_dir="./libriphone/feat/",
    phone_path="./libriphone",
    concat_nframes=concat_nframes,
)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# load model
model = Classifier(
    input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim
).to(device)
model.load_state_dict(torch.load(model_path))
# Make prediction.
test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(
            outputs, 1
        )  # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)


with open("prediction.csv", "w") as f:
    f.write("Id,Class\n")
    for i, y in enumerate(pred):
        f.write("{},{}\n".format(i, y))
