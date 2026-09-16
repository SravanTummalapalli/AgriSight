import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

MANIFEST_PATH = "dataset/leakage_free_dataset/manifest.csv"

OUTPUT_PATH = "dataset/leakage_free_dataset/fruit_manifest.csv"


FRUIT_CLASSES = [
    "Apple",
    "Banana",
    "Guava",
    "Lime",
    "Orange",
    "Pomegranate"
]


# ============================================================
# Load Manifest
# ============================================================

print("=" * 70)
print("AGRISIGHT - FRUIT CLASSIFIER DATASET PREPARATION")
print("=" * 70)

print("\nLoading manifest...")

df = pd.read_csv(MANIFEST_PATH)

print(f"Total images: {len(df)}")


# ============================================================
# Validate Columns
# ============================================================

required_columns = [
    "image_path",
    "fruit",
    "quality",
    "split",
    "group_id"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print(
        f"\n❌ Missing columns: {missing_columns}"
    )

    raise SystemExit


print("✅ Required columns found")


# ============================================================
# Validate Fruit Classes
# ============================================================

actual_fruits = sorted(
    df["fruit"].unique()
)

expected_fruits = sorted(
    FRUIT_CLASSES
)

print("\nFruit classes:")
print(actual_fruits)


if actual_fruits != expected_fruits:

    print("\n❌ Fruit classes do not match!")

    print(
        "Expected:",
        expected_fruits
    )

    print(
        "Found:",
        actual_fruits
    )

    raise SystemExit


print("✅ All six fruit classes found")


# ============================================================
# Validate Splits
# ============================================================

expected_splits = {
    "train",
    "val",
    "test"
}

actual_splits = set(
    df["split"].unique()
)

if actual_splits != expected_splits:

    print("\n❌ Unexpected dataset splits!")

    print(
        "Expected:",
        expected_splits
    )

    print(
        "Found:",
        actual_splits
    )

    raise SystemExit


print("✅ Train / Val / Test splits found")


# ============================================================
# Validate Files
# ============================================================

print("\nChecking image files...")

missing_files = []

for image_path in df["image_path"]:

    if not Path(image_path).is_file():

        missing_files.append(image_path)


if missing_files:

    print(
        f"\n❌ Missing files: {len(missing_files)}"
    )

    print("\nFirst few:")

    for path in missing_files[:10]:

        print(path)

    raise SystemExit


print("✅ All image files exist")


# ============================================================
# Validate Group Leakage
# ============================================================

print("\nChecking group leakage...")

group_split_counts = (
    df.groupby("group_id")["split"]
    .nunique()
)

leaking_groups = (
    group_split_counts[
        group_split_counts > 1
    ]
)

if len(leaking_groups) > 0:

    print(
        f"\n❌ Group leakage detected: "
        f"{len(leaking_groups)} groups"
    )

    raise SystemExit


print("✅ No group leakage")


# ============================================================
# Create Fruit Manifest
# ============================================================

fruit_df = df[
    [
        "image_path",
        "fruit",
        "split",
        "group_id"
    ]
].copy()


# ============================================================
# Save
# ============================================================

fruit_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\n✅ Fruit manifest saved:"
    f"\n   {OUTPUT_PATH}"
)


# ============================================================
# Display Distribution
# ============================================================

print("\n" + "=" * 70)
print("FRUIT DISTRIBUTION")
print("=" * 70)

distribution = (
    fruit_df
    .groupby(["fruit", "split"])
    .size()
    .unstack(fill_value=0)
)


print(distribution)


# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 70)
print("DATASET READY")
print("=" * 70)

print(
    f"\nTotal images: {len(fruit_df)}"
)

print(
    f"Training images: "
    f"{len(fruit_df[fruit_df['split'] == 'train'])}"
)

print(
    f"Validation images: "
    f"{len(fruit_df[fruit_df['split'] == 'val'])}"
)

print(
    f"Test images: "
    f"{len(fruit_df[fruit_df['split'] == 'test'])}"
)

print("\nFruit classes:")

for fruit in FRUIT_CLASSES:

    count = (
        fruit_df["fruit"] == fruit
    ).sum()

    print(
        f"  {fruit:<15} {count}"
    )

print("\n✅ Ready for fruit classifier training.")