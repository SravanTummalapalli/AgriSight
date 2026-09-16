import os
import pandas as pd
import numpy as np
import torch

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_large
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

MANIFEST = "dataset/quality_dataset/manifest.csv"

MODEL_PATH = "quality_model_best.pth"

IMAGE_SIZE = 224

BATCH_SIZE = 32

CLASS_NAMES = [
    "Good",
    "Bad",
    "Mixed"
]

CLASS_TO_INDEX = {
    "Good": 0,
    "Bad": 1,
    "Mixed": 2
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
print("AGRISIGHT QUALITY MODEL EVALUATION")
print("=" * 70)

print(f"\nDevice: {device}")


# ============================================================
# Transform
# ============================================================

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


test_df = df[
    df["split"] == "test"
].copy()


print(
    f"Test images: {len(test_df)}"
)


# ============================================================
# Dataset + DataLoader
# ============================================================

test_dataset = FruitQualityDataset(
    test_df,
    transform=eval_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Create model
# ============================================================

print("\nLoading model...")

model = mobilenet_v3_large(
    weights=None
)


input_features = (
    model.classifier[-1].in_features
)


model.classifier[-1] = torch.nn.Linear(
    input_features,
    3
)


# ============================================================
# Load trained weights
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(device)

model.eval()


print(
    f"Loaded: {MODEL_PATH}"
)


# ============================================================
# Predictions
# ============================================================

print("\nRunning predictions...")

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


all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


# ============================================================
# Overall Accuracy
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print("\n")
print("=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("Confusion Matrix:")
print()

print(
    f"{'Actual / Predicted':<20}"
    f"{'Good':>10}"
    f"{'Bad':>10}"
    f"{'Mixed':>10}"
)

print("-" * 50)


for i, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{class_name:<20}"
        f"{cm[i, 0]:>10}"
        f"{cm[i, 1]:>10}"
        f"{cm[i, 2]:>10}"
    )


# ============================================================
# Save confusion matrix image
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "AgriSight Quality Model - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(CLASS_NAMES)
)

plt.xticks(
    tick_marks,
    CLASS_NAMES
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)


# Add numbers

for i in range(
    len(CLASS_NAMES)
):

    for j in range(
        len(CLASS_NAMES)
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "quality_confusion_matrix.png",
    dpi=200
)

plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    "quality_confusion_matrix.png"
)


# ============================================================
# Fruit-wise evaluation
# ============================================================

print("\n")
print("=" * 70)
print("FRUIT-WISE PERFORMANCE")
print("=" * 70)


fruits = sorted(
    test_df["fruit"].unique()
)


fruit_results = []


for fruit in fruits:

    # Get indices belonging to this fruit
    fruit_mask = (
        test_df["fruit"].values
        == fruit
    )


    fruit_labels = all_labels[
        fruit_mask
    ]

    fruit_predictions = (
        all_predictions[
            fruit_mask
        ]
    )


    fruit_accuracy = accuracy_score(
        fruit_labels,
        fruit_predictions
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            fruit_labels,
            fruit_predictions,
            labels=[0, 1, 2],
            zero_division=0
        )
    )


    fruit_results.append({

        "Fruit": fruit,

        "Images": len(fruit_labels),

        "Accuracy": fruit_accuracy,

        "Good_F1": f1[0],

        "Bad_F1": f1[1],

        "Mixed_F1": f1[2]

    })


print()

print(
    f"{'Fruit':<15}"
    f"{'Images':>10}"
    f"{'Accuracy':>12}"
    f"{'Good F1':>12}"
    f"{'Bad F1':>12}"
    f"{'Mixed F1':>12}"
)

print("-" * 75)


for result in fruit_results:

    print(
        f"{result['Fruit']:<15}"
        f"{result['Images']:>10}"
        f"{result['Accuracy'] * 100:>11.2f}%"
        f"{result['Good_F1'] * 100:>11.2f}%"
        f"{result['Bad_F1'] * 100:>11.2f}%"
        f"{result['Mixed_F1'] * 100:>11.2f}%"
    )


# ============================================================
# Misclassified images
# ============================================================

print("\n")
print("=" * 70)
print("MISCLASSIFIED IMAGES")
print("=" * 70)


wrong_indices = np.where(
    all_predictions != all_labels
)[0]


print(
    f"\nTotal misclassified: "
    f"{len(wrong_indices)}"
)


if len(wrong_indices) > 0:

    print("\nFirst 20 mistakes:\n")

    print(
        f"{'Image':<50}"
        f"{'Fruit':<15}"
        f"{'Actual':<12}"
        f"{'Predicted':<12}"
    )

    print("-" * 90)


    for index in wrong_indices[:20]:

        row = test_df.iloc[index]

        actual = CLASS_NAMES[
            all_labels[index]
        ]

        predicted = CLASS_NAMES[
            all_predictions[index]
        ]


        filename = os.path.basename(
            row["image_path"]
        )


        print(
            f"{filename[:48]:<50}"
            f"{row['fruit']:<15}"
            f"{actual:<12}"
            f"{predicted:<12}"
        )


# ============================================================
# Final summary
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(
    f"\nOverall test accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Misclassified images: "
    f"{len(wrong_indices)}"
)

print(
    "\nSaved file:"
)

print(
    "quality_confusion_matrix.png"
)

print("=" * 70)