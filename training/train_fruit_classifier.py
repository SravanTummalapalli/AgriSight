import pandas as pd
import torch
import time

from PIL import Image
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# ============================================================
# Configuration
# ============================================================

MANIFEST_PATH = "dataset/leakage_free_dataset/fruit_manifest.csv"

MODEL_OUTPUT = "fruit_model_leakage_free_best.pth"

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

NUM_WORKERS = 0


# ============================================================
# Fruit Classes
# ============================================================

FRUIT_CLASSES = [
    "Apple",
    "Banana",
    "Guava",
    "Lime",
    "Orange",
    "Pomegranate"
]

CLASS_TO_INDEX = {
    fruit: index
    for index, fruit in enumerate(FRUIT_CLASSES)
}


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
print("AGRISIGHT - FRUIT CLASSIFIER TRAINING")
print("=" * 70)

print(f"\nDevice: {device}")


# ============================================================
# Load Manifest
# ============================================================

print("\nLoading manifest...")

df = pd.read_csv(MANIFEST_PATH)

print(f"Total images: {len(df)}")


# ============================================================
# Dataset
# ============================================================

class FruitDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["image_path"]

        fruit = row["fruit"]

        label = CLASS_TO_INDEX[fruit]


        # Load image
        image = Image.open(image_path).convert("RGB")


        # Transform
        if self.transform:

            image = self.transform(image)


        return image, label


# ============================================================
# Data Augmentation
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Create Splits
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print("\nDataset sizes:")

print(f"Train: {len(train_df)}")

print(f"Val:   {len(val_df)}")

print(f"Test:  {len(test_df)}")


# ============================================================
# Create Datasets
# ============================================================

train_dataset = FruitDataset(
    train_df,
    transform=train_transform
)

val_dataset = FruitDataset(
    val_df,
    transform=eval_transform
)

test_dataset = FruitDataset(
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
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# Calculate Class Weights
# ============================================================

class_counts = (
    train_df["fruit"]
    .value_counts()
)


print("\nTraining class counts:")

for fruit in FRUIT_CLASSES:

    print(
        f"{fruit:<15}: "
        f"{class_counts[fruit]}"
    )


# ------------------------------------------------------------
# Inverse-frequency class weights
# ------------------------------------------------------------

total_samples = len(train_df)

num_classes = len(FRUIT_CLASSES)


class_weights = []

for fruit in FRUIT_CLASSES:

    count = class_counts[fruit]

    weight = total_samples / (
        num_classes * count
    )

    class_weights.append(weight)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


print("\nClass weights:")

for fruit, weight in zip(
    FRUIT_CLASSES,
    class_weights
):

    print(
        f"{fruit:<15}: "
        f"{weight:.4f}"
    )


class_weights = class_weights.to(device)


# ============================================================
# Load Pretrained MobileNetV3
# ============================================================

print("\nLoading pretrained MobileNetV3-Large...")

weights = MobileNet_V3_Large_Weights.DEFAULT

model = mobilenet_v3_large(
    weights=weights
)


# ============================================================
# Replace Classifier
# ============================================================

input_features = (
    model.classifier[-1].in_features
)


model.classifier[-1] = torch.nn.Linear(
    input_features,
    num_classes
)


model = model.to(device)


print("✅ Model ready")


# ============================================================
# Loss Function
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
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Learning Rate Scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# Training Function
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


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


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )


    return epoch_loss, epoch_accuracy


# ============================================================
# Validation Function
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


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


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )


    return epoch_loss, epoch_accuracy


# ============================================================
# Training Loop
# ============================================================

best_val_loss = float("inf")


print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    start_time = time.time()


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy = (
        validate()
    )


    scheduler.step(
        val_loss
    )


    current_lr = optimizer.param_groups[0]["lr"]


    epoch_time = (
        time.time()
        - start_time
    )


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss:      {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:  {train_accuracy:.4f}"
    )

    print(
        f"Val Loss:        {val_loss:.4f}"
    )

    print(
        f"Val Accuracy:    {val_accuracy:.4f}"
    )

    print(
        f"Learning Rate:   {current_lr:.6f}"
    )

    print(
        f"Time:            {epoch_time:.1f}s"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "fruit_classes": FRUIT_CLASSES,

                "image_size": IMAGE_SIZE,

                "best_val_loss": best_val_loss
            },
            MODEL_OUTPUT
        )


        print(
            f"✅ Best model saved → "
            f"{MODEL_OUTPUT}"
        )


# ============================================================
# Load Best Model
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


checkpoint = torch.load(
    MODEL_OUTPUT,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print("✅ Best model loaded")


# ============================================================
# Test Evaluation
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)


correct = 0

total = 0


# Per-class statistics
class_correct = {
    fruit: 0
    for fruit in FRUIT_CLASSES
}

class_total = {
    fruit: 0
    for fruit in FRUIT_CLASSES
}


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(images)


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        for label, prediction in zip(
            labels,
            predictions
        ):

            label_index = int(
                label.item()
            )

            prediction_index = int(
                prediction.item()
            )


            fruit_name = FRUIT_CLASSES[
                label_index
            ]


            class_total[
                fruit_name
            ] += 1


            if label_index == prediction_index:

                class_correct[
                    fruit_name
                ] += 1


test_accuracy = (
    correct / total
)


print(
    f"\nOverall Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


print("\nPer-fruit accuracy:")

for fruit in FRUIT_CLASSES:

    accuracy = (
        class_correct[fruit]
        /
        class_total[fruit]
    )


    print(
        f"{fruit:<15}: "
        f"{accuracy * 100:.2f}% "
        f"({class_correct[fruit]}/"
        f"{class_total[fruit]})"
    )


# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 70)
print("FRUIT CLASSIFIER TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"\nModel saved at:"
    f"\n{MODEL_OUTPUT}"
)

print("\nFruit classes:")

for index, fruit in enumerate(
    FRUIT_CLASSES
):

    print(
        f"  {index}: {fruit}"
    )

print("\n✅ Training complete.")