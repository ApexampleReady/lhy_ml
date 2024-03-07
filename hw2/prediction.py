import torch
import random
import os
import numpy as np
import tqdm as tqdm


# Useful functions
def preprocess_data(
    split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, random_seed=1213
):
    class_num = 41  # NOTE: pre-computed, should not need change

    if split == "train" or split == "val":
        mode = "train"
    elif split == "test":
        mode = "test"
    else:
        raise ValueError("Invalid 'split' argument for dataset: PhoneDataset!")

    label_dict = {}
    if mode == "train":
        for line in open(os.path.join(phone_path, f"{mode}_labels.txt")).readlines():
            line = line.strip("\n").split(" ")
            label_dict[line[0]] = [int(p) for p in line[1:]]

        # split training and validation data
        usage_list = open(os.path.join(phone_path, "train_split.txt")).readlines()
        random.seed(random_seed)
        random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = (
            usage_list[:train_len] if split == "train" else usage_list[train_len:]
        )

    elif mode == "test":
        usage_list = open(os.path.join(phone_path, "test_split.txt")).readlines()

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
    if mode == "train":
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for _, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f"{fname}.pt"))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode == "train":
            label = torch.LongTensor(label_dict[fname])

        X[idx : idx + cur_len, :] = feat
        if mode == "train":
            y[idx : idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode == "train":
        y = y[:idx]

    print(f"[INFO] {split} set")
    print(X.shape)
    if mode == "train":
        print(y.shape)
        return X, y
    else:
        return X


import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        # TODO: apply batch normalization and dropout for strong baseline.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html (batch normalization)
        #       https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html (dropout)
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
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


# load data
test_X = preprocess_data(
    split="test",
    feat_dir="./libriphone/feat",
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

"""Make prediction."""

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

"""Write prediction to a CSV file.

After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.
"""
print(pred.shape)
counter = 0
with open("prediction.csv", "w") as f:
    f.write("Id,Class\n")
    for i, y in enumerate(pred):
        f.write("{},{}\n".format(i, y))
        counter += 1
