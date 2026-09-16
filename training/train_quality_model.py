import os
import pandas as pd
import numpy as np
import torch

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# Configuration
# ============================================================

MANIFEST = "dataset/leakage_free_dataset/manifest.csv"

MODEL_OUTPUT = "quality_model_leakage_free_best.pth"

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.0001

NUM_CLASSES = 3

CLASS_NAMES = [
    "Good",
    "Bad",
    "Mixed"
]


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("=" * 70)
print("AGRISIGHT QUALITY MODEL TRAINING")
print("=" * 70)

print(f"\nDevice: {device}")


# ============================================================
# Class mapping
# ============================================================

CLASS_TO_INDEX = {
    "Good": 0,
    "Bad": 1,
    "Mixed": 2
}


# ============================================================
# Image transformations
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# Dataset
# ============================================================

class FruitQualityDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["image_path"]

        quality = row["quality"]


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(
                image
            )


        label = CLASS_TO_INDEX[
            quality
        ]


        return image, label


# ============================================================
# Load manifest
# ============================================================

print("\nLoading manifest...")

df = pd.read_csv(
    MANIFEST
)


train_df = df[
    df["split"] == "train"
].copy()


val_df = df[
    df["split"] == "val"
].copy()


test_df = df[
    df["split"] == "test"
].copy()


print(
    f"Train      : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Test       : {len(test_df)}"
)


# ============================================================
# Create datasets
# ============================================================

train_dataset = FruitQualityDataset(
    train_df,
    transform=train_transform
)


val_dataset = FruitQualityDataset(
    val_df,
    transform=eval_transform
)


test_dataset = FruitQualityDataset(
    test_df,
    transform=eval_transform
)


# ============================================================
# Create DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Load pretrained MobileNetV3
# ============================================================

print("\nLoading MobileNetV3-Large...")

weights = MobileNet_V3_Large_Weights.DEFAULT

model = mobilenet_v3_large(
    weights=weights
)


# ============================================================
# Replace classifier
# ============================================================

input_features = (
    model.classifier[-1].in_features
)

model.classifier[-1] = torch.nn.Linear(
    input_features,
    NUM_CLASSES
)


model = model.to(device)


# ============================================================
# Class weights
# ============================================================

class_weights = torch.tensor(
    [
        0.558888,
        0.956622,
        6.046358
    ],
    dtype=torch.float32
).to(device)


print("\nClass weights:")

for name, weight in zip(
    CLASS_NAMES,
    class_weights
):

    print(
        f"{name:<10}: "
        f"{weight.item():.4f}"
    )


# ============================================================
# Loss function
# ============================================================

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# Learning rate scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# Training function
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_labels = []


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


    epoch_loss = (
        running_loss
        / len(train_dataset)
    )


    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return epoch_loss, epoch_accuracy


# ============================================================
# Validation function
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )


            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )


    epoch_loss = (
        running_loss
        / len(val_dataset)
    )


    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# Training loop
# ============================================================

best_val_loss = float("inf")


print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    print(
        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS}"
    )


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy = (
        validate()
    )


    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    print(
        f"Train Loss     : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy : "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val Loss       : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy   : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Learning Rate  : "
        f"{current_lr:.7f}"
    )


    # Save best model

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "image_size":
                    IMAGE_SIZE
            },
            MODEL_OUTPUT
        )


        print(
            "✅ Best model saved!"
        )


# ============================================================
# Load best model
# ============================================================

print("\nLoading best model...")

checkpoint = torch.load(
    MODEL_OUTPUT,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# Final test evaluation
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


model.eval()

all_predictions = []

all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(
            images
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions
            .cpu()
            .numpy()
        )


        all_labels.extend(
            labels
            .numpy()
        )


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
        zero_division=0
    )
)


print(
    f"\nOverall Accuracy: "
    f"{accuracy:.4f}"
)


print("\nClass Performance:")

print(
    f"{'Class':<12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Support':>12}"
)

print("-" * 60)


for i, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{class_name:<12}"
        f"{precision[i]:>12.4f}"
        f"{recall[i]:>12.4f}"
        f"{f1[i]:>12.4f}"
        f"{support[i]:>12}"
    )


print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest model saved to:"
)

print(
    MODEL_OUTPUT
)