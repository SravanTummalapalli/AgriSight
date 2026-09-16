import pandas as pd

MANIFEST_PATH = "dataset/leakage_free_dataset/manifest.csv"

df = pd.read_csv(MANIFEST_PATH)

# Use TRAINING DATA ONLY
train_df = df[df["split"] == "train"]

counts = train_df["quality"].value_counts()

total = len(train_df)
num_classes = len(counts)

weights = total / (num_classes * counts)

print("=" * 60)
print("LEAKAGE-FREE DATASET CLASS WEIGHTS")
print("=" * 60)

print("\nTraining class counts:")

for quality in ["Good", "Bad", "Mixed"]:
    print(f"{quality:10s}: {counts[quality]}")

print("\nClass weights:")

for quality in ["Good", "Bad", "Mixed"]:
    print(f"{quality:10s}: {weights[quality]:.6f}")

print("\n" + "=" * 60)