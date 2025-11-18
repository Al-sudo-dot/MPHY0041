import os
import numpy as np
import nibabel as nib

# --- PATH TO YOUR DATASET ---
DATASET_PATH = "/Users/anastasia/Desktop/prostate cancer/Dataset_prostate.nosync"
OUTPUT_PATH = "./data/prostate-data"

os.makedirs(OUTPUT_PATH, exist_ok=True)

zones_dir = os.path.join(DATASET_PATH, "zonesTr")
images_dir = os.path.join(DATASET_PATH, "imagesTr")

# --- get all mask filenames ---
zone_files = sorted([f for f in os.listdir(zones_dir) if f.lower().endswith(".nii")])

# extract 3-digit patient IDs
patient_ids = sorted([f.split()[0] for f in zone_files])

print(f"Found {len(patient_ids)} patients")

index = 0

for pid in patient_ids:

    # -----------------------------
    # LOAD PROSTATE MASK (label)
    # -----------------------------
    mask_file = [f for f in zone_files if f.startswith(pid)][0]
    mask_path = os.path.join(zones_dir, mask_file)
    mask = nib.load(mask_path).get_fdata()

    # -----------------------------
    # LOAD IMAGE (T2 = *_0000.nii)
    # -----------------------------
    image_file = f"{pid}_0000.nii"

    if not os.path.exists(os.path.join(images_dir, image_file)):
        print(f"Missing T2 image for patient {pid}, skipping")
        continue

    image_path = os.path.join(images_dir, image_file)
    image = nib.load(image_path).get_fdata()

    # -----------------------------
    # DOWNSAMPLE TO MATCH TRAINING CODE
    # -----------------------------
    image = image[::2, ::2, ::2]
    mask = mask[::2, ::2, ::2]

    # -----------------------------
    # SAVE AS NUMPY FILES
    # -----------------------------
    np.save(os.path.join(OUTPUT_PATH, f"image_train{index:02d}.npy"), image)
    np.save(os.path.join(OUTPUT_PATH, f"label_train{index:02d}.npy"), mask)

    index += 1
    print(f"Saved patient {pid} as index {index-1}")

print("Done! Files saved in", OUTPUT_PATH)
