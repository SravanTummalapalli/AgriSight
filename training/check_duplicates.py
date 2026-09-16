import pandas as pd
from PIL import Image
import imagehash
from collections import defaultdict


# ============================================================
# Configuration
# ============================================================

MANIFEST = "dataset/quality_dataset/manifest.csv"

HASH_THRESHOLD = 5


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(MANIFEST)

print("=" * 70)
print("AGRISIGHT DUPLICATE / NEAR-DUPLICATE CHECK")
print("=" * 70)

print(
    f"\nTotal images: {len(df)}"
)


# ============================================================
# Calculate perceptual hashes
# ============================================================

print("\nCalculating perceptual hashes...")

hashes = []

failed = []


for index, row in df.iterrows():

    path = row["image_path"]

    try:

        with Image.open(path) as image:

            image = image.convert("RGB")

            # Perceptual hash
            image_hash = imagehash.phash(
                image
            )

            hashes.append(
                str(image_hash)
            )

    except Exception as error:

        hashes.append(None)

        failed.append(
            (path, str(error))
        )


df["hash"] = hashes


print(
    f"Hashing completed."
)

print(
    f"Failed images: {len(failed)}"
)


# ============================================================
# Exact perceptual hash duplicates
# ============================================================

print("\n")
print("=" * 70)
print("EXACT PERCEPTUAL HASH DUPLICATES")
print("=" * 70)


hash_groups = defaultdict(list)


for index, row in df.iterrows():

    image_hash = row["hash"]

    if image_hash is not None:

        hash_groups[image_hash].append(
            index
        )


duplicate_groups = []

for image_hash, indices in hash_groups.items():

    if len(indices) > 1:

        duplicate_groups.append(
            (image_hash, indices)
        )


print(
    f"\nDuplicate groups found: "
    f"{len(duplicate_groups)}"
)


# ============================================================
# Compare dataset splits
# ============================================================

def compare_splits(
    split_a,
    split_b
):

    print("\n")
    print(
        "=" * 70
    )

    print(
        f"{split_a.upper()} ↔ "
        f"{split_b.upper()}"
    )

    print(
        "=" * 70
    )


    df_a = df[
        df["split"] == split_a
    ]

    df_b = df[
        df["split"] == split_b
    ]


    # Create hash lookup for split B

    hash_lookup = defaultdict(list)


    for index, row in df_b.iterrows():

        if row["hash"] is not None:

            hash_lookup[
                row["hash"]
            ].append(index)


    exact_matches = []


    for index, row in df_a.iterrows():

        image_hash = row["hash"]

        if image_hash in hash_lookup:

            for matching_index in hash_lookup[
                image_hash
            ]:

                exact_matches.append(
                    (
                        index,
                        matching_index
                    )
                )


    print(
        f"\nExact hash matches: "
        f"{len(exact_matches)}"
    )


    # --------------------------------------------------------
    # Near duplicate comparison
    # --------------------------------------------------------

    # Convert hashes into objects

    hash_objects_a = []

    hash_objects_b = []


    for index, row in df_a.iterrows():

        if row["hash"] is not None:

            hash_objects_a.append(
                (
                    index,
                    imagehash.hex_to_hash(
                        row["hash"]
                    )
                )
            )


    for index, row in df_b.iterrows():

        if row["hash"] is not None:

            hash_objects_b.append(
                (
                    index,
                    imagehash.hex_to_hash(
                        row["hash"]
                    )
                )
            )


    near_matches = []


    # Compare hashes

    for index_a, hash_a in hash_objects_a:

        for index_b, hash_b in hash_objects_b:

            distance = (
                hash_a - hash_b
            )


            if (
                distance > 0
                and distance <= HASH_THRESHOLD
            ):

                near_matches.append(
                    (
                        index_a,
                        index_b,
                        distance
                    )
                )


    print(
        f"Near-duplicate matches "
        f"(distance 1-{HASH_THRESHOLD}): "
        f"{len(near_matches)}"
    )


    # --------------------------------------------------------
    # Display examples
    # --------------------------------------------------------

    if len(exact_matches) > 0:

        print(
            "\nExamples of exact matches:"
        )

        for index_a, index_b in exact_matches[:10]:

            row_a = df.loc[index_a]

            row_b = df.loc[index_b]

            print(
                f"\nA: {row_a['image_path']}"
            )

            print(
                f"B: {row_b['image_path']}"
            )

            print(
                f"Fruit: "
                f"{row_a['fruit']} / "
                f"{row_b['fruit']}"
            )

            print(
                f"Quality: "
                f"{row_a['quality']} / "
                f"{row_b['quality']}"
            )


    if len(near_matches) > 0:

        print(
            "\nExamples of near matches:"
        )

        for index_a, index_b, distance in (
            near_matches[:10]
        ):

            row_a = df.loc[index_a]

            row_b = df.loc[index_b]

            print(
                f"\nDistance: {distance}"
            )

            print(
                f"A: {row_a['image_path']}"
            )

            print(
                f"B: {row_b['image_path']}"
            )

            print(
                f"Fruit: "
                f"{row_a['fruit']} / "
                f"{row_b['fruit']}"
            )

            print(
                f"Quality: "
                f"{row_a['quality']} / "
                f"{row_b['quality']}"
            )


    return (
        len(exact_matches),
        len(near_matches)
    )


# ============================================================
# Run comparisons
# ============================================================

train_val = compare_splits(
    "train",
    "val"
)

train_test = compare_splits(
    "train",
    "test"
)

val_test = compare_splits(
    "val",
    "test"
)


# ============================================================
# Final summary
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DUPLICATE CHECK SUMMARY")
print("=" * 70)

print(
    f"\nTrain ↔ Validation"
)

print(
    f"Exact matches : {train_val[0]}"
)

print(
    f"Near matches  : {train_val[1]}"
)


print(
    f"\nTrain ↔ Test"
)

print(
    f"Exact matches : {train_test[0]}"
)

print(
    f"Near matches  : {train_test[1]}"
)


print(
    f"\nValidation ↔ Test"
)

print(
    f"Exact matches : {val_test[0]}"
)

print(
    f"Near matches  : {val_test[1]}"
)


print("\n" + "=" * 70)
print("DUPLICATE CHECK COMPLETED")
print("=" * 70)