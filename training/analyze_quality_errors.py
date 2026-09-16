import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_large
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# Configuration
# ============================================================

MANIFEST = "dataset/leakage_free_dataset/manifest.csv"

MODEL_PATH = "quality_model_leakage_free_best.pth"

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
print("AGRISIGHT QUALITY MODEL ERROR ANALYSIS")
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


        return image, label, index


# ============================================================
# Load manifest
# ============================================================

print("\nLoading manifest...")

df = pd.read_csv(
    MANIFEST
)


test_df = df[
    df["split"] == "test"
].copy().reset_index(drop=True)


print(
    f"Test images: {len(test_df)}"
)


# ============================================================
# Dataset / DataLoader
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
# Load model
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
    len(CLASS_NAMES)
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(device)

model.eval()


print("✅ Model loaded")


# ============================================================
# Predictions
# ============================================================

print("\nRunning predictions...")

all_predictions = []

all_labels = []

all_confidences = []

all_indices = []


with torch.no_grad():

    for images, labels, indices in test_loader:

        images = images.to(device)

        outputs = model(
            images
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        confidences, predictions = (
            torch.max(
                probabilities,
                dim=1
            )
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


        all_confidences.extend(
            confidences
            .cpu()
            .numpy()
        )


        all_indices.extend(
            indices
            .numpy()
        )


# ============================================================
# Overall metrics
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
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)


# ============================================================
# Classification report
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2]
)


print("Confusion Matrix:")
print()

print(
    f"{'Actual / Pred':<15}"
    f"{'Good':>10}"
    f"{'Bad':>10}"
    f"{'Mixed':>10}"
)

print("-" * 45)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<15}"
        f"{cm[i, 0]:>10}"
        f"{cm[i, 1]:>10}"
        f"{cm[i, 2]:>10}"
    )


# ============================================================
# Save confusion matrix
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm
)

plt.title(
    "AgriSight - Leakage-Free Test Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)


for i in range(len(CLASS_NAMES)):

    for j in range(len(CLASS_NAMES)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "quality_leakage_free_confusion_matrix.png",
    dpi=200
)

plt.close()


print(
    "\n✅ Confusion matrix saved:"
)

print(
    "quality_leakage_free_confusion_matrix.png"
)


# ============================================================
# Build results dataframe
# ============================================================

results = test_df.copy()

results["actual_index"] = all_labels

results["predicted_index"] = all_predictions

results["predicted_quality"] = [
    CLASS_NAMES[i]
    for i in all_predictions
]

results["confidence"] = all_confidences

results["correct"] = (
    results["actual_index"]
    ==
    results["predicted_index"]
)


# ============================================================
# Misclassified images
# ============================================================

errors = results[
    results["correct"] == False
].copy()


errors = errors.sort_values(
    "confidence",
    ascending=False
)


print("\n")
print("=" * 70)
print("MISCLASSIFIED IMAGES")
print("=" * 70)

print(
    f"\nTotal misclassified: "
    f"{len(errors)}"
)


if len(errors) > 0:

    print()

    for number, (_, row) in enumerate(
        errors.iterrows(),
        start=1
    ):

        print(
            f"{number}. "
            f"{row['fruit']} | "
            f"Actual: {row['quality']} | "
            f"Predicted: {row['predicted_quality']} | "
            f"Confidence: "
            f"{row['confidence'] * 100:.2f}%"
        )

        print(
            f"   {row['image_path']}"
        )


# ============================================================
# Save all errors to CSV
# ============================================================

error_columns = [
    "image_path",
    "fruit",
    "quality",
    "predicted_quality",
    "confidence",
    "group_id"
]


errors[
    error_columns
].to_csv(
    "quality_misclassified_test_images.csv",
    index=False
)


print(
    "\n✅ Error list saved:"
)

print(
    "quality_misclassified_test_images.csv"
)


# ============================================================
# Fruit-wise accuracy
# ============================================================

print("\n")
print("=" * 70)
print("FRUIT-WISE TEST PERFORMANCE")
print("=" * 70)

fruit_results = []

for fruit in sorted(
    results["fruit"].unique()
):

    fruit_df = results[
        results["fruit"] == fruit
    ]

    fruit_accuracy = (
        fruit_df["correct"].mean()
    )

    fruit_results.append(
        {
            "fruit": fruit,
            "total": len(fruit_df),
            "correct": int(
                fruit_df["correct"].sum()
            ),
            "incorrect": int(
                (~fruit_df["correct"]).sum()
            ),
            "accuracy": fruit_accuracy
        }
    )

    print(
        f"{fruit:<15}"
        f"{fruit_accuracy * 100:>8.2f}% "
        f"("
        f"{int(fruit_df['correct'].sum())}/"
        f"{len(fruit_df)}"
        f")"
    )


pd.DataFrame(
    fruit_results
).to_csv(
    "quality_fruit_wise_performance.csv",
    index=False
)


print(
    "\n✅ Fruit-wise results saved:"
)

print(
    "quality_fruit_wise_performance.csv"
)


# ============================================================
# Confidence analysis
# ============================================================

print("\n")
print("=" * 70)
print("ERROR CONFIDENCE ANALYSIS")
print("=" * 70)

if len(errors) > 0:

    print(
        f"\nHighest-confidence wrong prediction:"
    )

    highest = errors.iloc[0]

    print(
        f"Fruit      : {highest['fruit']}"
    )

    print(
        f"Actual     : {highest['quality']}"
    )

    print(
        f"Predicted  : {highest['predicted_quality']}"
    )

    print(
        f"Confidence : "
        f"{highest['confidence'] * 100:.2f}%"
    )

    print(
        f"\nLowest-confidence wrong prediction:"
    )

    lowest = errors.iloc[-1]

    print(
        f"Fruit      : {lowest['fruit']}"
    )

    print(
        f"Actual     : {lowest['quality']}"
    )

    print(
        f"Predicted  : {lowest['predicted_quality']}"
    )

    print(
        f"Confidence : "
        f"{lowest['confidence'] * 100:.2f}%"
    )


# ============================================================
# Completion
# ============================================================

print("\n")
print("=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)
