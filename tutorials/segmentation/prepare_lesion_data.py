import os
import numpy as np
import csv
import nibabel as nib
from sklearn.model_selection import train_test_split

# ====== PATHS ======
DATASET_ROOT = "/Users/anastasia/Desktop/prostate cancer/Dataset_prostate.nosync"
IMAGES_DIR   = os.path.join(DATASET_ROOT, "imagesTr")
LABELS_DIR   = os.path.join(DATASET_ROOT, "labelsTr")   # segmentation labels
OUTPUT_DIR   = "./data/lesion-data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- mapping file ----
MAP_PATH = os.path.join(OUTPUT_DIR, "index_to_pid.csv")

# ====== IMPORTANT: LABEL CHOICE ======
# From your sanity check, label 1 looks organ-sized (prostate/zones),
# label 2 looks focal (lesion). So we take LESION_LABEL = 2.
LESION_LABEL = 2

# If True, skip patients with no lesion voxels (no label==LESION_LABEL).
# Recommended initially to make training easier.
SKIP_NO_LESION = True

# If True, delete old saved npy files first to avoid index mix-ups
CLEAR_OUTPUT_FIRST = True


# ------------ Helper functions -------------

def load_nifti(path):
    return nib.load(path).get_fdata()

def find_t2(pid: str) -> str:
    t2_file = f"{pid}.nii"
    t2_path = os.path.join(IMAGES_DIR, t2_file)
    if os.path.exists(t2_path):
        return t2_path
    raise FileNotFoundError(f"T2 not found for patient {pid}: expected {t2_file}")

def find_mask(pid: str) -> str:
    mask_path = os.path.join(LABELS_DIR, f"{pid}.nii")
    if os.path.exists(mask_path):
        return mask_path
    raise FileNotFoundError(f"Mask not found for patient {pid}: expected {pid}.nii in labelsTr")

def clear_output_dir():
    to_delete = [f for f in os.listdir(OUTPUT_DIR)
                 if f.endswith(".npy") or f == os.path.basename(MAP_PATH)]
    for f in to_delete:
        try:
            os.remove(os.path.join(OUTPUT_DIR, f))
        except FileNotFoundError:
            pass
    if os.path.exists(MAP_PATH):
        os.remove(MAP_PATH)


# ------------ Build patient list -------------

patients = [
    f.replace(".nii", "")
    for f in os.listdir(IMAGES_DIR)
    if f.endswith(".nii") and "_" not in f
]
patients = sorted(patients)

print(f"Found {len(patients)} T2 patients:", patients[:10])

if len(patients) == 0:
    print("❌ No usable T2 files found! Expected <ID>.nii files in imagesTr.")
    raise SystemExit


# ------------ Train / Test / Holdout Split --------------

train_ids, temp_ids = train_test_split(
    patients, test_size=0.30, random_state=42
)

test_ids, holdout_ids = train_test_split(
    temp_ids, test_size=(1/3), random_state=42
)

print(f"\nSplit summary:")
print(f"  Train:   {len(train_ids)} patients")
print(f"  Test:    {len(test_ids)} patients")
print(f"  Holdout: {len(holdout_ids)} patients\n")


# ------------ Process & Save ----------------

def next_index(split: str) -> int:
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"image_{split}")]
    return len(existing)

def write_map_row(split: str, idx: int, pid: str):
    write_header = not os.path.exists(MAP_PATH)
    with open(MAP_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["split", "idx", "pid"])
        w.writerow([split, f"{idx:03d}", pid])

def process_and_save(pid: str, split: str):
    t2_path   = find_t2(pid)
    mask_path = find_mask(pid)

    img = load_nifti(t2_path).astype(np.float32)
    mask_raw = load_nifti(mask_path)

    # ---- lesion-only mask (CRITICAL FIX) ----
    mask = (mask_raw == LESION_LABEL).astype(np.float32)

    lesion_voxels = int(mask.sum())
    if lesion_voxels == 0:
        msg = f"⚠️ {pid}: no lesion voxels for label {LESION_LABEL}"
        if SKIP_NO_LESION:
            print(msg + " → skipping")
            return
        else:
            print(msg + " → keeping as negative")

    # ---- normalize image ----
    # Use nonzero voxels if present (common for MRI with zero-padding/background)
    nonzero = img > 0
    img_mean = img[nonzero].mean() if np.any(nonzero) else img.mean()
    img_std  = img[nonzero].std()  if np.any(nonzero) else img.std()
    if img_std > 0:
        img = (img - img_mean) / img_std

    idx = next_index(split)

    write_map_row(split, idx, pid)

    np.save(os.path.join(OUTPUT_DIR, f"image_{split}{idx:03d}.npy"), img)
    np.save(os.path.join(OUTPUT_DIR, f"label_{split}{idx:03d}.npy"), mask)

    print(f"✓ Saved {pid} → {split}{idx:03d} (lesion voxels={lesion_voxels})")


if __name__ == "__main__":

    if CLEAR_OUTPUT_FIRST:
        print("Clearing previous outputs in:", OUTPUT_DIR)
        clear_output_dir()

    print("\nProcessing TRAIN patients…")
    for pid in train_ids:
        process_and_save(pid, "train")

    print("\nProcessing TEST patients…")
    for pid in test_ids:
        process_and_save(pid, "test")

    print("\nProcessing HOLDOUT patients…")
    for pid in holdout_ids:
        process_and_save(pid, "holdout")

    print("\n✔ Done! Files saved in ./data/lesion-data")
    print(f"Used lesion label value = {LESION_LABEL}")
    print(f"SKIP_NO_LESION = {SKIP_NO_LESION}")
