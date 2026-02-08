import os
import numpy as np
import csv
import nibabel as nib
from sklearn.model_selection import train_test_split

# ====== PATHS ======
DATASET_ROOT = "/Users/anastasia/Desktop/prostate cancer/Dataset_prostate.nosync"
IMAGES_DIR   = os.path.join(DATASET_ROOT, "imagesTr")
LABELS_DIR   = os.path.join(DATASET_ROOT, "labelsTr")   # lesion labels
ZONES_DIR    = os.path.join(DATASET_ROOT, "zonesTr")    # prostate/zones masks (WEIRD FILENAMES)

OUTPUT_DIR   = "./data/lesion-roi-data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAP_PATH = os.path.join(OUTPUT_DIR, "index_to_pid.csv")

# ====== LABEL CHOICE ======
LESION_LABEL = 2

# keep negatives is fine for ROI training
SKIP_NO_LESION = False

# delete old outputs
CLEAR_OUTPUT_FIRST = True

# ROI padding around prostate bbox
PAD_XY = 16
PAD_Z  = 4


# ------------ Helper functions -------------

def load_nifti(path):
    return nib.load(path).get_fdata()

def find_standard_file(folder, pid: str) -> str:
    """Find <pid>.nii in folder (imagesTr/labelsTr use this)."""
    path = os.path.join(folder, f"{pid}.nii")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Missing file for patient {pid}: {path}")

def find_zones_file(pid: str) -> str:
    """
    zonesTr files are named like: '119 19.47.44.nii'
    i.e. they start with '<pid><space>'
    """
    # 1) try normal <pid>.nii (in case some exist)
    p = os.path.join(ZONES_DIR, f"{pid}.nii")
    if os.path.exists(p):
        return p

    # 2) try "pid + space" prefix match
    prefix = f"{pid} "
    for f in os.listdir(ZONES_DIR):
        if f.startswith(prefix) and f.endswith(".nii"):
            return os.path.join(ZONES_DIR, f)

    raise FileNotFoundError(f"Missing zones file for patient {pid} in zonesTr")

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

def bbox_from_mask(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1
    return [x0, x1, y0, y1, z0, z1]

def pad_bbox(b, shape, pad_xy=16, pad_z=4):
    x0, x1, y0, y1, z0, z1 = b
    H, W, D = shape
    x0 = max(0, x0 - pad_xy); x1 = min(H, x1 + pad_xy)
    y0 = max(0, y0 - pad_xy); y1 = min(W, y1 + pad_xy)
    z0 = max(0, z0 - pad_z ); z1 = min(D, z1 + pad_z )
    return [x0, x1, y0, y1, z0, z1]

def crop3d(vol, b):
    x0, x1, y0, y1, z0, z1 = b
    return vol[x0:x1, y0:y1, z0:z1]

def normalize(img):
    nonzero = img > 0
    mean = img[nonzero].mean() if np.any(nonzero) else img.mean()
    std  = img[nonzero].std()  if np.any(nonzero) else img.std()
    if std > 0:
        img = (img - mean) / std
    return img.astype(np.float32)

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


# ------------ Build patient list (must have image+label+zones) -------------

patients_all = sorted([
    f.replace(".nii", "")
    for f in os.listdir(IMAGES_DIR)
    if f.endswith(".nii") and "_" not in f
])

patients = []
missing_zones = 0

for pid in patients_all:
    # imagesTr and labelsTr are standard names
    img_ok = os.path.exists(os.path.join(IMAGES_DIR, f"{pid}.nii"))
    lab_ok = os.path.exists(os.path.join(LABELS_DIR, f"{pid}.nii"))
    if not (img_ok and lab_ok):
        continue

    # zones are weird, so try find_zones_file
    try:
        _ = find_zones_file(pid)
        patients.append(pid)
    except FileNotFoundError:
        missing_zones += 1

patients = sorted(patients)

print(f"Found {len(patients_all)} total T2 patients.")
print(f"Using {len(patients)} patients with image+label+zones.")
print(f"Skipped {missing_zones} patients missing zones masks.")
print("First 10 usable:", patients[:10])

if len(patients) == 0:
    raise SystemExit("❌ No patients have matching zones masks — cannot build ROI dataset.")


# ------------ Train / Test / Holdout Split --------------

train_ids, temp_ids = train_test_split(patients, test_size=0.30, random_state=42)
test_ids, holdout_ids = train_test_split(temp_ids, test_size=(1/3), random_state=42)

print(f"\nSplit summary:")
print(f"  Train:   {len(train_ids)} patients")
print(f"  Test:    {len(test_ids)} patients")
print(f"  Holdout: {len(holdout_ids)} patients\n")


# ------------ Process & Save (ROI crop) ----------------

def process_and_save(pid: str, split: str):
    t2_path     = find_standard_file(IMAGES_DIR, pid)
    lesion_path = find_standard_file(LABELS_DIR, pid)
    zones_path  = find_zones_file(pid)

    img = load_nifti(t2_path).astype(np.float32)
    lesion_raw = load_nifti(lesion_path)
    zones_raw  = load_nifti(zones_path)

    # ROI is anything > 0 in zones mask
    roi_mask = (zones_raw > 0).astype(np.uint8)

    b = bbox_from_mask(roi_mask)
    if b is None:
        print(f"⚠️ {pid}: EMPTY zones mask → skipping")
        return

    b = pad_bbox(b, img.shape, pad_xy=PAD_XY, pad_z=PAD_Z)

    # lesion-only binary mask
    lesion = (lesion_raw == LESION_LABEL).astype(np.float32)
    if lesion.sum() == 0 and SKIP_NO_LESION:
        print(f"⚠️ {pid}: no lesion voxels (label {LESION_LABEL}) → skipping")
        return

    img_c = crop3d(img, b)
    lesion_c = crop3d(lesion, b)

    img_c = normalize(img_c)

    idx = next_index(split)
    write_map_row(split, idx, pid)

    np.save(os.path.join(OUTPUT_DIR, f"image_{split}{idx:03d}.npy"), img_c)
    np.save(os.path.join(OUTPUT_DIR, f"label_{split}{idx:03d}.npy"), lesion_c)

    print(f"✓ Saved {pid} → {split}{idx:03d} | crop={img_c.shape} | lesion voxels ROI={int(lesion_c.sum())}")


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

    print("\n✔ Done! Files saved in ./data/lesion-roi-data")
    print(f"Used lesion label value = {LESION_LABEL}")
    print(f"SKIP_NO_LESION = {SKIP_NO_LESION}")
