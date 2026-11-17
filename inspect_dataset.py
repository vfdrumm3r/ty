
import os
import glob
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_FOLDER = "dataset"

EXPECTED_FRAMES = 20
EXPECTED_FEATURES = 126

# ==========================================
# DATASET INSPECTION
# ==========================================

def inspect_dataset():

    if not os.path.exists(DATASET_FOLDER):
        print("Dataset folder not found!")
        return

    label_folders = [
        folder for folder in os.listdir(DATASET_FOLDER)
        if os.path.isdir(os.path.join(DATASET_FOLDER, folder))
    ]

    if not label_folders:
        print("No label folders found!")
        return

    total_samples = 0
    valid_samples = 0
    invalid_samples = 0

    print("\n========== DATASET REPORT ==========\n")

    for label in sorted(label_folders):

        folder_path = os.path.join(DATASET_FOLDER, label)

        files = glob.glob(
            os.path.join(folder_path, "*.npz")
        )

        print(f"\nLabel: {label}")
        print(f"Samples found: {len(files)}")

        label_valid = 0
        label_invalid = 0

        for file_path in files:

            total_samples += 1

            try:
                with np.load(file_path, allow_pickle=False) as data:

                    sequence = data["sequence"]
                    saved_label = str(data["label"].item())

                    correct_shape = sequence.shape == (
                        EXPECTED_FRAMES,
                        EXPECTED_FEATURES
                    )

                    valid_numbers = np.isfinite(sequence).all()

                    correct_label = saved_label == label

                    if (
                        correct_shape
                        and valid_numbers
                        and correct_label
                    ):

                        valid_samples += 1
                        label_valid += 1

                    else:

                        invalid_samples += 1
                        label_invalid += 1

                        print("\n  INVALID:", os.path.basename(file_path))

                        if not correct_shape:
                            print("  Incorrect shape:", sequence.shape)

                        if not valid_numbers:
                            print("  Contains invalid numerical values")

                        if not correct_label:
                            print("  Folder label:", label)
                            print("  Saved label:", saved_label)

            except Exception as error:

                invalid_samples += 1
                label_invalid += 1

                print("\n  ERROR:", os.path.basename(file_path))
                print(" ", error)

        print(f"Valid samples: {label_valid}")
        print(f"Invalid samples: {label_invalid}")

    # ======================================
    # FINAL SUMMARY
    # ======================================

    print("\n========== FINAL SUMMARY ==========")

    print("Total samples:", total_samples)
    print("Valid samples:", valid_samples)
    print("Invalid samples:", invalid_samples)

    if total_samples > 0:
        print(
            "Dataset validity:",
            round(valid_samples / total_samples * 100, 2),
            "%"
        )

    print("\n===================================\n")


if __name__ == "__main__":
    inspect_dataset()
