import os
import shutil
import random
from collections import Counter

# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = "dataset"
OUTPUT_DIR = "dataset/quality_dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(RANDOM_SEED)


# ============================================================
# Fruit name detection
# ============================================================

FRUITS = [
    "Apple",
    "Banana",
    "Guava",
    "Lime",
    "Orange",
    "Pomegranate"
]


def get_fruit(folder_name):
    """
    Identify fruit from the folder name.
    """

    folder_lower = folder_name.lower()

    for fruit in FRUITS:

        if folder_lower.startswith(fruit.lower()):
            return fruit

    return None


# ============================================================
# Quality detection
# ============================================================

def get_quality(path):

    path_lower = path.lower()

    if "good quality" in path_lower:
        return "Good"

    if "bad quality" in path_lower:
        return "Bad"

    if "mixed quality" in path_lower:
        return "Mixed"

    return None


# ============================================================
# Find all images
# ============================================================

images = []

print("\nScanning dataset...\n")

for root, directories, files in os.walk(SOURCE_DIR):

    # Don't scan our output directory again
    if OUTPUT_DIR in root:
        continue

    for filename in files:

        extension = os.path.splitext(filename)[1].lower()

        if extension not in IMAGE_EXTENSIONS:
            continue

        file_path = os.path.join(root, filename)

        # Determine quality
        quality = get_quality(file_path)

        # Determine fruit from folder names
        fruit = None

        for part in root.split(os.sep):

            detected_fruit = get_fruit(part)

            if detected_fruit:
                fruit = detected_fruit
                break

        if fruit is None or quality is None:
            print("⚠️ Could not classify:")
            print(file_path)
            continue

        images.append({
            "path": file_path,
            "fruit": fruit,
            "quality": quality
        })


# ============================================================
# Dataset summary before splitting
# ============================================================

print("=" * 60)
print("DATASET FOUND")
print("=" * 60)

print(f"\nTotal usable images: {len(images)}")


fruit_counts = Counter()
quality_counts = Counter()
fruit_quality_counts = Counter()

for item in images:

    fruit_counts[item["fruit"]] += 1
    quality_counts[item["quality"]] += 1

    fruit_quality_counts[
        (item["fruit"], item["quality"])
    ] += 1


print("\nImages by fruit:")

for fruit in FRUITS:

    print(
        f"{fruit:<15} : "
        f"{fruit_counts[fruit]}"
    )


print("\nImages by quality:")

for quality in ["Good", "Bad", "Mixed"]:

    print(
        f"{quality:<15} : "
        f"{quality_counts[quality]}"
    )


print("\nFruit + Quality:")

print(
    f"{'Fruit':<15}"
    f"{'Good':>10}"
    f"{'Bad':>10}"
    f"{'Mixed':>10}"
)

print("-" * 45)

for fruit in FRUITS:

    print(
        f"{fruit:<15}"
        f"{fruit_quality_counts[(fruit, 'Good')]:>10}"
        f"{fruit_quality_counts[(fruit, 'Bad')]:>10}"
        f"{fruit_quality_counts[(fruit, 'Mixed')]:>10}"
    )


# ============================================================
# Group images by fruit + quality
# ============================================================

groups = {}

for item in images:

    key = (
        item["fruit"],
        item["quality"]
    )

    if key not in groups:
        groups[key] = []

    groups[key].append(item)


# ============================================================
# Create output directories
# ============================================================

splits = [
    "train",
    "val",
    "test"
]

qualities = [
    "Good",
    "Bad",
    "Mixed"
]


print("\nCreating directories...")

for split in splits:

    for quality in qualities:

        directory = os.path.join(
            OUTPUT_DIR,
            split,
            quality
        )

        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================
# Split and copy images
# ============================================================

split_counts = Counter()

print("\nSplitting dataset...\n")

for key, group in groups.items():

    fruit, quality = key

    # Shuffle
    random.shuffle(group)

    total = len(group)

    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(total * VAL_RATIO)

    train_images = group[:train_end]

    val_images = group[train_end:val_end]

    test_images = group[val_end:]


    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }


    for split_name, split_images in split_data.items():

        destination_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            quality
        )

        for index, item in enumerate(split_images):

            original_path = item["path"]

            filename = os.path.basename(
                original_path
            )

            # Add fruit name to avoid duplicate filenames
            new_filename = (
                f"{fruit}_{index}_"
                f"{filename}"
            )

            destination_path = os.path.join(
                destination_dir,
                new_filename
            )

            shutil.copy2(
                original_path,
                destination_path
            )

            split_counts[
                (split_name, quality)
            ] += 1


# ============================================================
# Final report
# ============================================================

print("\n")
print("=" * 60)
print("PREPARED DATASET")
print("=" * 60)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\n")

print(
    f"{'Split':<15}"
    f"{'Good':>10}"
    f"{'Bad':>10}"
    f"{'Mixed':>10}"
    f"{'Total':>10}"
)

print("-" * 60)

for split in splits:

    good = split_counts[(split, "Good")]
    bad = split_counts[(split, "Bad")]
    mixed = split_counts[(split, "Mixed")]

    total = good + bad + mixed

    print(
        f"{split:<15}"
        f"{good:>10}"
        f"{bad:>10}"
        f"{mixed:>10}"
        f"{total:>10}"
    )


print("\n" + "=" * 60)
print("Dataset preparation completed!")
print("=" * 60)

print("\nOriginal dataset was NOT modified.")
print("The prepared dataset is a separate copy.")