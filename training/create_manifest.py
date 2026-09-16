import os
import csv

# ============================================================
# Configuration
# ============================================================

DATASET_DIR = "dataset/quality_dataset"
OUTPUT_FILE = "dataset/quality_dataset/manifest.csv"

SPLITS = ["train", "val", "test"]

QUALITIES = ["Good", "Bad", "Mixed"]

FRUITS = [
    "Apple",
    "Banana",
    "Guava",
    "Lime",
    "Orange",
    "Pomegranate"
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# Detect fruit from filename
# ============================================================

def get_fruit(filename):

    for fruit in FRUITS:

        if filename.startswith(fruit + "_"):
            return fruit

    return None


# ============================================================
# Create manifest
# ============================================================

rows = []

print("\nCreating manifest...\n")

for split in SPLITS:

    for quality in QUALITIES:

        folder = os.path.join(
            DATASET_DIR,
            split,
            quality
        )

        if not os.path.exists(folder):

            print(f"⚠️ Folder not found: {folder}")
            continue

        for filename in os.listdir(folder):

            extension = os.path.splitext(filename)[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            fruit = get_fruit(filename)

            if fruit is None:

                print(
                    f"⚠️ Could not determine fruit: "
                    f"{filename}"
                )

                continue

            image_path = os.path.join(
                folder,
                filename
            )

            rows.append({
                "image_path": image_path,
                "fruit": fruit,
                "quality": quality,
                "split": split
            })


# ============================================================
# Write CSV
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_path",
            "fruit",
            "quality",
            "split"
        ]
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# Report
# ============================================================

print("=" * 60)
print("MANIFEST CREATED")
print("=" * 60)

print(f"\nTotal images: {len(rows)}")

print(f"\nManifest:")
print(OUTPUT_FILE)

print("\n")

# Count by split + quality

from collections import Counter

counts = Counter()

for row in rows:

    counts[
        (row["split"], row["quality"])
    ] += 1


print(
    f"{'Split':<15}"
    f"{'Good':>10}"
    f"{'Bad':>10}"
    f"{'Mixed':>10}"
    f"{'Total':>10}"
)

print("-" * 60)

for split in SPLITS:

    good = counts[(split, "Good")]
    bad = counts[(split, "Bad")]
    mixed = counts[(split, "Mixed")]

    total = good + bad + mixed

    print(
        f"{split:<15}"
        f"{good:>10}"
        f"{bad:>10}"
        f"{mixed:>10}"
        f"{total:>10}"
    )


print("\n" + "=" * 60)
print("Manifest creation completed!")
print("=" * 60)