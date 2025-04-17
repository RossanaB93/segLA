import os
import os.path as osp
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandAffined, EnsureTyped, EnsureChannelFirstd, AsDiscrete
)
from monai.inferers import sliding_window_inference
from sklearn.model_selection import KFold

# Directory setup
images_dir = "/mnt/data/dataset/images"
labels_dir = "/mnt/data/dataset/labels"
logdir = "/mnt/data/logs"
os.makedirs(logdir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get filenames
images = sorted(glob.glob(osp.join(images_dir, "*.nii.gz")))
labels = sorted(glob.glob(osp.join(labels_dir, "*.nii.gz")))
data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

# Transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4
    ),
    RandAffined(
        keys=["image", "label"], prob=0.3, rotate_range=(0.1, 0.1, 0.1),
        scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")
    ),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

# Cross-validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(data_dicts)):
    print(f"Fold {fold+1}/{num_folds}")
    train_files = [data_dicts[i] for i in train_idx]
    val_files = [data_dicts[i] for i in val_idx]

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    max_epochs = 100
    best_metric = -1
    best_metric_epoch = -1
    train_loss_list, val_dice_list = [], []

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = sliding_window_inference(inputs, (96, 96, 96), 1, model)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_loss_list.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            dice_metric.reset()
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 1, model)
                val_outputs = [AsDiscrete(argmax=True, to_onehot=2)(i) for i in decollate_batch(val_outputs)]
                val_labels = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)
            metric = dice_metric.aggregate().item()
            val_dice_list.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), osp.join(logdir, f"best_metric_model_fold{fold+1}.pth"))
                print("Saved new best model!")
            print(f"Validation Dice: {metric:.4f}, Best so far: {best_metric:.4f} at epoch {best_metric_epoch}")

    # Plotting metrics
    plt.figure("Training Metrics", (12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_dice_list, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()

    plt.savefig(osp.join(logdir, f"metrics_fold{fold+1}.png"))
    plt.close()

    print(f"Training complete for fold {fold+1}, best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
