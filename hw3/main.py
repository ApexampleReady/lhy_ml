from operator import truediv
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image
import timm
from sklearn.model_selection import KFold

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset

# This is for the progress bar.
from tqdm.auto import tqdm
import random


myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""# Transforms
Torchvision provides lots of useful utilities for image preprocessing, data *wrapping* as well as data augmentation.

Please refer to PyTorch official website for details about different transforms.
"""

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
transform_type = random.randint(1, 5)
amount_of_transform = random.randint(1, 4)
train_tfm = transforms.Compose(
    [
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize((128, 128)),
        *[
            transforms.RandomChoice(
                [
                    transforms.RandomRotation((-60, 60)),
                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                    transforms.RandomErasing(value="random"),
                    transforms.Grayscale(num_output_channels=3),
                ]
            )
            for _ in range(amount_of_transform)
        ],
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""


class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted(
            [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")]
        )
        if files != None:
            self.files = files

        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label

        return im, label


"""# Model"""


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]
            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]
            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]
            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]
            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


"""# Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
# model = Classifier().to(device)
model = timm.create_model("resnest14d", pretrained=False)
model.cuda()

_exp_name = "DA1+MS1"

load_model_from_disk = True

# The number of batch size.
batch_size = 64

# The number of training epochs.
n_epochs = 8

# The amount of folds to use
k_folds = 5

# If no improvement in 'patience' epochs, early stop.
patience = 8

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

"""# Dataloader"""
kfold = KFold(k_folds, shuffle=True)
dataset = FoodDataset("./dataset/", tfm=train_tfm)

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_subsampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_subsampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if load_model_from_disk:
        state_dict = torch.load(f"{_exp_name}_best.ckpt")
        model.load_state_dict(state_dict)

    """# Start Training"""

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0

    for epoch in range(n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()
            # print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(
            f"[ Train | Fold:{fold}/{k_folds} |{epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}"
        )

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            # break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(
            f"[ Valid | Fold:{fold}/{k_folds} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}"
        )

        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt", "a"):
                print(
                    f"[ Valid | Fold:{fold}/{k_folds} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best"
                )
        else:
            with open(f"./{_exp_name}_log.txt", "a"):
                print(
                    f"[ Valid | Fold:{fold}/{k_folds} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}"
                )

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch},fold {fold}, saving model")
            torch.save(
                model.state_dict(), f"{_exp_name}_best.ckpt"
            )  # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break

"""# Dataloader for test"""

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset("./test", tfm=test_tfm)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
)

"""# Testing and generate prediction CSV"""

model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data, _ in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()


# create test csv
def pad4(i):
    return "0" * (4 - len(str(i))) + str(i)


df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission.csv", index=False)

"""# Q1. Augmentation Implementation
## Implement augmentation by finishing train_tfm in the code with image size of your choice.
## Directly copy the following block and paste it on GradeScope after you finish the code
### Your train_tfm must be capable of producing 5+ different results when given an identical image multiple times.
### Your  train_tfm in the report can be different from train_tfm in your training code.

"""

train_tfm = transforms.Compose(
    [
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize((128, 128)),
        # You can add some transforms here.
        transforms.ToTensor(),
    ]
)

"""# Q2. Visual Representations Implementation
## Visualize the learned visual representations of the CNN model on the validation set by implementing t-SNE (t-distributed Stochastic Neighbor Embedding) on the output of both top & mid layers (You need to submit 2 images).

"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the trained model
# model = Classifier().to(device)
timm.create_model("resnest14d", pretrained=False)
state_dict = torch.load(f"{_exp_name}_best.ckpt")
model.load_state_dict(state_dict)
model.eval()

print(model)

# Load the vaildation set defined by TA
valid_set = FoodDataset("./valid", tfm=test_tfm)
valid_loader = DataLoader(
    valid_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True
)

# Extract the representations for the specific layer of model
index = (
    ...
)  # You should find out the index of layer which is defined as "top" or 'mid' layer of your model.
features = []
labels = []
for batch in tqdm(valid_loader):
    imgs, lbls = batch
    with torch.no_grad():
        logits = model.cnn[:index](imgs.to(device))
        logits = logits.view(logits.size()[0], -1)
    labels.extend(lbls.cpu().numpy())
    logits = np.squeeze(logits.cpu().numpy())
    features.extend(logits)

features = np.array(features)
colors_per_class = cm.rainbow(np.linspace(0, 1, 11))

# Apply t-SNE to the features
features_tsne = TSNE(n_components=2, init="pca", random_state=42).fit_transform(
    features
)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    plt.scatter(
        features_tsne[labels == label, 0],
        features_tsne[labels == label, 1],
        label=label,
        s=5,
    )
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
labels = [0]
for label in np.unique(labels):
    plt.scatter(
        features_tsne[labels == label, 0],
        features_tsne[labels == label, 1],
        label=label,
        s=5,
    )
plt.legend()
plt.show()
