import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

# ====== CHANGE THIS ONLY IF YOUR DATA MOVES ======
DATASET_ROOT = "/Users/anastasia/Desktop/prostate cancer/Dataset_prostate.nosync"
IMAGES_DIR = os.path.join(DATASET_ROOT, "imagesTr")   # T2 = <ID>.nii
ZONES_DIR  = os.path.join(DATASET_ROOT, "zonesTr")    # Mask = <ID> something.nii
OUTPUT_DIR = "./data/prostate-data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------ Helper functions -------------

def load_nifti(path):
    return nib.load(path).get_fdata()

def find_t2(pid):
    """Your T2 is EXACTLY pid.nii (e.g. 067.nii)."""
    t2_file = f"{pid}.nii"
    t2_path = os.path.join(IMAGES_DIR, t2_file)
    if os.path.exists(t2_path):
        return t2_path
    raise FileNotFoundError(f"T2 not found for patient {pid}: expected {t2_file}")

def find_mask(pid):
    """
    Mask files look like: 067 12.45.22.nii
    """
    for f in os.listdir(ZONES_DIR):
        if f.startswith(pid + " "):
            return os.path.join(ZONES_DIR, f)
    raise FileNotFoundError(f"Mask not found for patient {pid}")

# ------------ Load patient list -------------

# T2 files are ONLY the ones matching:  ### <ID>.nii ###
patients = [f.replace(".nii", "") for f in os.listdir(IMAGES_DIR)
            if f.endswith(".nii") and "_" not in f]   # removes 067_0001.nii etc

patients = sorted(patients)

print(f"Found {len(patients)} T2 patients:", patients[:10])

if len(patients) == 0:
    print("❌ No usable T2 files found! Expected <ID>.nii files.")
    raise SystemExit

# ------------ Train/Test Split --------------

train_ids, test_ids = train_test_split(patients, test_size=0.20, random_state=42)

# ------------ Process & Save ----------------

def process_and_save(pid, split):
    t2_path = find_t2(pid)
    mask_path = find_mask(pid)

    img = load_nifti(t2_path)
    mask = load_nifti(mask_path)

    mask = (mask > 0).astype(np.float32)

    idx = len([f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"image_{split}")])

    np.save(os.path.join(OUTPUT_DIR, f"image_{split}{idx:03d}.npy"), img.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, f"label_{split}{idx:03d}.npy"), mask.astype(np.float32))

    print(f"✓ Saved {pid} → {split}{idx:03d}")

# ------------ Run processing ----------------

print("\nProcessing TRAIN patients…")
for pid in train_ids:
    process_and_save(pid, "train")

print("\nProcessing TEST patients…")
for pid in test_ids:
    process_and_save(pid, "test")

print("\n✔ Done! Files saved in ./data/prostate-data")
