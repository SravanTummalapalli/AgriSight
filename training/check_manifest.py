import os
import pandas as pd
from PIL import Image

MANIFEST_PATH = "dataset/leakage_free_dataset/manifest.csv"

print("=" * 70)
print("LEAKAGE-FREE DATASET MANIFEST CHECK")
print("=" * 70)

df = pd.read_csv(MANIFEST_PATH)

print(f"\nManifest rows: {len(df)}")

# ---------------------------------------------------------
# 1. Check required columns
# ---------------------------------------------------------

required_columns = ["image_path", "fruit", "quality", "split", "group_id"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print(f"\n❌ Missing columns: {missing_columns}")
    exit()

print("✅ Required columns present")

# ---------------------------------------------------------
# 2. Check missing files
# ---------------------------------------------------------

missing_files = []

for path in df["image_path"]:
    if not os.path.exists(path):
        missing_files.append(path)

print(f"\nMissing files: {len(missing_files)}")

if missing_files:
    print("❌ Missing files detected")
    print(missing_files[:10])
else:
    print("✅ All files exist")

# ---------------------------------------------------------
# 3. Check image loading
# ---------------------------------------------------------

failed_images = []

for path in df["image_path"]:
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        failed_images.append(path)

print(f"Failed image loads: {len(failed_images)}")

if failed_images:
    print("❌ Some images cannot be opened")
    print(failed_images[:10])
else:
    print("✅ All images are valid")

# ---------------------------------------------------------
# 4. Check split distribution
# ---------------------------------------------------------

print("\nSplit distribution:")

print(
    df.groupby(["split", "quality"])
      .size()
      .unstack(fill_value=0)
)

# ---------------------------------------------------------
# 5. Check fruit distribution
# ---------------------------------------------------------

print("\nFruit distribution:")

print(
    df.groupby(["split", "fruit"])
      .size()
      .unstack(fill_value=0)
)

# ---------------------------------------------------------
# 6. Check group leakage
# ---------------------------------------------------------

group_split_counts = (
    df.groupby("group_id")["split"]
      .nunique()
)

leaking_groups = group_split_counts[
    group_split_counts > 1
]

print("\nGroup leakage check:")
print(f"Groups appearing in multiple splits: {len(leaking_groups)}")

if len(leaking_groups) == 0:
    print("✅ NO GROUP LEAKAGE")
else:
    print("❌ GROUP LEAKAGE DETECTED")

# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print("\n" + "=" * 70)

if (
    len(missing_files) == 0
    and len(failed_images) == 0
    and len(leaking_groups) == 0
):
    print("✅ MANIFEST CHECK PASSED")
    print("=" * 70)
else:
    print("❌ MANIFEST CHECK FAILED")
    print("=" * 70)