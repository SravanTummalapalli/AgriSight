import os
import random
import shutil
from collections import defaultdict, Counter

import pandas as pd
from PIL import Image
import imagehash


# ============================================================
# Configuration
# ============================================================

SOURCE_MANIFEST = "dataset/quality_dataset/manifest.csv"

OUTPUT_DIR = "dataset/leakage_free_dataset"

OUTPUT_MANIFEST = (
    "dataset/leakage_free_dataset/manifest.csv"
)

HASH_THRESHOLD = 5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

QUALITIES = [
    "Good",
    "Bad",
    "Mixed"
]

FRUITS = [
    "Apple",
    "Banana",
    "Guava",
    "Lime",
    "Orange",
    "Pomegranate"
]


# ============================================================
# Load manifest
# ============================================================

print("=" * 70)
print("AGRISIGHT LEAKAGE-FREE DATASET CREATION")
print("=" * 70)

df = pd.read_csv(
    SOURCE_MANIFEST
)

print(
    f"\nTotal images: {len(df)}"
)


# ============================================================
# Calculate perceptual hashes
# ============================================================

print("\nCalculating perceptual hashes...")

hashes = []

for index, row in df.iterrows():

    try:

        with Image.open(
            row["image_path"]
        ) as image:

            image_hash = imagehash.phash(
                image.convert("RGB")
            )

            hashes.append(image_hash)

    except Exception as error:

        print(
            f"⚠️ Failed: {row['image_path']}"
        )

        print(error)

        hashes.append(None)


df["phash"] = hashes


print(
    "Failed hashes:",
    df["phash"].isna().sum()
)


# ============================================================
# Union-Find
# ============================================================

parent = list(range(len(df)))


def find(x):

    if parent[x] != x:

        parent[x] = find(parent[x])

    return parent[x]


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:

        parent[root_b] = root_a


# ============================================================
# Exact duplicate grouping
# ============================================================

print("\nGrouping exact duplicates...")

hash_to_indices = defaultdict(list)

for index, row in df.iterrows():

    if row["phash"] is not None:

        hash_to_indices[
            str(row["phash"])
        ].append(index)


exact_groups = 0

for indices in hash_to_indices.values():

    if len(indices) > 1:

        exact_groups += 1

        first = indices[0]

        for other in indices[1:]:

            union(
                first,
                other
            )


print(
    f"Exact duplicate groups: "
    f"{exact_groups}"
)


# ============================================================
# Near duplicate grouping
# ============================================================

print("\nGrouping near-duplicates...")

category_items = defaultdict(list)

for index, row in df.iterrows():

    if row["phash"] is not None:

        key = (
            row["fruit"],
            row["quality"]
        )

        category_items[key].append(
            (
                index,
                row["phash"]
            )
        )


near_pairs = 0


for key, items in category_items.items():

    for i in range(len(items)):

        index_a, hash_a = items[i]

        for j in range(i + 1, len(items)):

            index_b, hash_b = items[j]

            distance = (
                hash_a - hash_b
            )

            if (
                distance > 0
                and distance <= HASH_THRESHOLD
            ):

                near_pairs += 1

                union(
                    index_a,
                    index_b
                )


print(
    f"Near-duplicate pairs grouped: "
    f"{near_pairs}"
)


# ============================================================
# Build groups
# ============================================================

print("\nCreating image groups...")

components = defaultdict(list)

for index in range(len(df)):

    root = find(index)

    components[root].append(index)


groups = []

for group_id, indices in enumerate(
    components.values()
):

    group_df = df.iloc[
        indices
    ]

    fruit = group_df[
        "fruit"
    ].mode()[0]

    quality = group_df[
        "quality"
    ].mode()[0]

    groups.append({

        "group_id": group_id,

        "indices": indices,

        "fruit": fruit,

        "quality": quality,

        "size": len(indices)

    })


print(
    f"Total image groups: "
    f"{len(groups)}"
)


# ============================================================
# Group statistics
# ============================================================

sizes = [
    group["size"]
    for group in groups
]

print("\nGroup size statistics:")

print(
    f"Smallest group : {min(sizes)}"
)

print(
    f"Largest group  : {max(sizes)}"
)

print(
    f"Average group  : "
    f"{sum(sizes) / len(sizes):.2f}"
)


# ============================================================
# Organize groups by fruit + quality
# ============================================================

category_groups = defaultdict(list)

for group in groups:

    key = (
        group["fruit"],
        group["quality"]
    )

    category_groups[key].append(
        group
    )


# ============================================================
# Split groups
# ============================================================

print("\nAssigning groups to splits...")

assignments = {}


for category, group_list in (
    category_groups.items()
):

    fruit, quality = category

    random.shuffle(
        group_list
    )


    total_images = sum(
        group["size"]
        for group in group_list
    )


    target_train = (
        total_images
        * TRAIN_RATIO
    )

    target_val = (
        total_images
        * VAL_RATIO
    )


    train_images = 0
    val_images = 0


    for group in group_list:

        size = group["size"]


        # Fill train first
        if train_images < target_train:

            assignments[
                group["group_id"]
            ] = "train"

            train_images += size


        # Then validation
        elif val_images < target_val:

            assignments[
                group["group_id"]
            ] = "val"

            val_images += size


        # Everything remaining → test
        else:

            assignments[
                group["group_id"]
            ] = "test"


# ============================================================
# Create output folders
# ============================================================

print("\nCreating output directories...")

for split in [
    "train",
    "val",
    "test"
]:

    for quality in QUALITIES:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                quality
            ),
            exist_ok=True
        )


# ============================================================
# Copy images
# ============================================================

print("\nCopying images...")

output_rows = []

split_counts = Counter()


for group in groups:

    group_id = group["group_id"]

    split = assignments[
        group_id
    ]

    fruit = group["fruit"]

    quality = group["quality"]


    destination_dir = os.path.join(
        OUTPUT_DIR,
        split,
        quality
    )


    for image_number, index in enumerate(
        group["indices"]
    ):

        row = df.iloc[index]

        source_path = row[
            "image_path"
        ]


        original_filename = os.path.basename(
            source_path
        )


        new_filename = (
            f"{fruit}_"
            f"G{group_id}_"
            f"{image_number}_"
            f"{original_filename}"
        )


        destination_path = os.path.join(
            destination_dir,
            new_filename
        )


        shutil.copy2(
            source_path,
            destination_path
        )


        output_rows.append({

            "image_path":
                destination_path,

            "fruit":
                fruit,

            "quality":
                quality,

            "split":
                split,

            "group_id":
                group_id

        })


        split_counts[
            (split, quality)
        ] += 1


# ============================================================
# Save manifest
# ============================================================

output_df = pd.DataFrame(
    output_rows
)

output_df.to_csv(
    OUTPUT_MANIFEST,
    index=False
)


# ============================================================
# Final split report
# ============================================================

print("\n")
print("=" * 70)
print("LEAKAGE-FREE DATASET")
print("=" * 70)

print(
    f"\nTotal images: "
    f"{len(output_df)}"
)

print(
    f"Total groups: "
    f"{len(groups)}"
)


print("\n")

print(
    f"{'Split':<15}"
    f"{'Good':>10}"
    f"{'Bad':>10}"
    f"{'Mixed':>10}"
    f"{'Total':>10}"
)

print("-" * 60)


for split in [
    "train",
    "val",
    "test"
]:

    good = split_counts[
        (split, "Good")
    ]

    bad = split_counts[
        (split, "Bad")
    ]

    mixed = split_counts[
        (split, "Mixed")
    ]

    total = (
        good
        + bad
        + mixed
    )


    print(
        f"{split:<15}"
        f"{good:>10}"
        f"{bad:>10}"
        f"{mixed:>10}"
        f"{total:>10}"
    )


# ============================================================
# Fruit distribution by split
# ============================================================

print("\nFruit distribution by split:")

fruit_split_counts = (
    output_df
    .groupby(
        ["split", "fruit"]
    )
    .size()
)


for split in [
    "train",
    "val",
    "test"
]:

    print(
        f"\n{split.upper()}:"
    )

    for fruit in FRUITS:

        count = fruit_split_counts.get(
            (split, fruit),
            0
        )

        print(
            f"  {fruit:<15}: {count}"
        )


# ============================================================
# Leakage verification
# ============================================================

print("\n")
print("=" * 70)
print("GROUP LEAKAGE VERIFICATION")
print("=" * 70)


group_split_counts = (
    output_df
    .groupby("group_id")["split"]
    .nunique()
)


leaking_groups = (
    group_split_counts[
        group_split_counts > 1
    ]
)


print(
    f"\nGroups appearing in multiple splits: "
    f"{len(leaking_groups)}"
)


if len(leaking_groups) == 0:

    print(
        "✅ NO GROUP LEAKAGE DETECTED"
    )

else:

    print(
        "❌ GROUP LEAKAGE DETECTED"
    )


# ============================================================
# Final
# ============================================================

print("\n")
print("=" * 70)
print("LEAKAGE-FREE SPLIT COMPLETED")
print("=" * 70)

print(
    "\nNew dataset:"
)

print(
    OUTPUT_DIR
)

print(
    "\nNew manifest:"
)

print(
    OUTPUT_MANIFEST
)

print(
    "\nOriginal dataset was NOT modified."
)

print("=" * 70)