# ==== Prepare Prostate Data for UNet Training ====
import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

# ====== MODIFY THIS PATH ONLY ======
DATASET_ROOT = "/Users/anastasia/Desktop/prostate cancer/Dataset_prostate.nosync"

IMAGES_DIR = os.path.join(DATASET_ROOT, "imagesTr")
ZONES_DIR  = os.path.join(DATASET_ROOT, "zonesTr")
OUTPUT_DIR = "./data/prostate-data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_nifti(path):
    return nib.load(path).get_fdata()

def minmax_norm(vol):
    vmin, vmax = vol.min(), vol.max()
    return (vol - vmin) / (vmax - vmin + 1e-8)


def find_mask(pid):
    """Find prostate mask matching 'PID ' prefix."""
    for f in os.listdir(ZONES_DIR):
        if f.startswith(pid + " "):
            return os.path.join(ZONES_DIR, f)
    return None


# ------------------------------------------------------------
# Identify valid patients
# ------------------------------------------------------------
print("Scanning imagesTr for T2 files (*.nii)...")
all_files = sorted(os.listdir(IMAGES_DIR))

# T2 images are *_0001.nii
t2_files = [f for f in all_files if f.endswith("_0001.nii")]

patient_ids = [f.split("_")[0] for f in t2_files]
print(f"Found {len(patient_ids)} T2 patients")

# Safety check
if len(patient_ids) == 0:
    raise RuntimeError("❌ No T2 images found. Check your dataset paths.")

# ------------------------------------------------------------
# Train/Test Split
# ------------------------------------------------------------
train_ids, test_ids = train_test_split(patient_ids, test_size=0.20, random_state=42)


# ------------------------------------------------------------
# Main processing function
# ------------------------------------------------------------
def process(pid, split):
    t2_path = os.path.join(IMAGES_DIR, f"{pid}_0001.nii")
    mask_path = find_mask(pid)

    if not os.path.exists(t2_path):
        print(f"⚠ Missing T2 for {pid}, skipping.")
        return

    if mask_path is None:
        print(f"⚠ Missing mask for {pid}, skipping.")
        return

    img = load_nifti(t2_path)
    mask = load_nifti(mask_path)

    img = minmax_norm(img)
    mask = (mask > 0.5).astype(np.float32)

    idx = len([f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"image_{split}")])

    np.save(os.path.join(OUTPUT_DIR, f"image_{split}{idx:03d}.npy"), img.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, f"label_{split}{idx:03d}.npy"), mask.astype(np.float32))

    print(f"✓ Saved {pid} → {split}{idx:03d}")


# ------------------------------------------------------------
# Run processing
# ------------------------------------------------------------
print("\nProcessing TRAIN patients…")
for pid in train_ids:
    process(pid, "train")

print("\nProcessing TEST patients…")
for pid in test_ids:
    process(pid, "test")

print("\n✔ DONE — NPY files saved in ./data/prostate-data")
