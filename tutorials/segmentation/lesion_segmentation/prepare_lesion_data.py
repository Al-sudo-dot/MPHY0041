import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

# ====== CHANGE THIS ONLY IF YOUR DATA MOVES ======
DATASET_ROOT = "/Users/anastasia/Desktop/prostate cancer/Dataset_prostate.nosync"

IMAGES_DIR = os.path.join(DATASET_ROOT, "imagesTr")   # pid.nii and/or pid_0001.nii, pid_0002.nii
LABELS_DIR = os.path.join(DATASET_ROOT, "labelsTr")   # lesion label: pid.nii or pid.nii.gz
ZONES_DIR  = os.path.join(DATASET_ROOT, "zonesTr")    # prostate mask: pid.nii(.gz) or "pid something.nii"

OUTPUT_DIR = "./data/lesion-data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------ Helper functions -------------

def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)

def zscore_nonzero(img: np.ndarray) -> np.ndarray:
    """
    Same idea as your prostate prep:
    normalise using nonzero voxels if possible.
    """
    img = img.astype(np.float32)
    mask = img != 0
    if np.any(mask):
        mean = img[mask].mean()
        std  = img[mask].std()
    else:
        mean = img.mean()
        std  = img.std()
    if std > 0:
        img = (img - mean) / std
    return img.astype(np.float32)

def to_zyx(vol_xyz: np.ndarray) -> np.ndarray:
    """
    nibabel often returns (X,Y,Z). Your training code assumes (Z,Y,X).
    So convert (X,Y,Z) -> (Z,Y,X).
    """
    if vol_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol_xyz.shape}")
    return np.transpose(vol_xyz, (2, 1, 0)).astype(np.float32)

def find_modalities(pid: str):
    """
    Your dataset is inconsistent:
      - some cases have pid_0001.nii and pid_0002.nii
      - some cases also have pid.nii

    We build a consistent 3-channel input:
      ch0 = pid.nii if exists else pid_0001.nii (fallback)
      ch1 = pid_0001.nii if exists else zeros
      ch2 = pid_0002.nii if exists else zeros
    """
    plain = os.path.join(IMAGES_DIR, f"{pid}.nii")
    m1    = os.path.join(IMAGES_DIR, f"{pid}_0001.nii")
    m2    = os.path.join(IMAGES_DIR, f"{pid}_0002.nii")

    plain_exists = os.path.exists(plain)
    m1_exists    = os.path.exists(m1)
    m2_exists    = os.path.exists(m2)

    if plain_exists:
        ch0 = plain
    elif m1_exists:
        ch0 = m1
    else:
        raise FileNotFoundError(
            f"No usable image found for {pid}. Expected {pid}.nii or {pid}_0001.nii"
        )

    ch1 = m1 if m1_exists else None
    ch2 = m2 if m2_exists else None

    return ch0, ch1, ch2

def find_lesion_label(pid: str) -> str:
    for ext in [".nii", ".nii.gz"]:
        p = os.path.join(LABELS_DIR, pid + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Lesion label not found for {pid} in labelsTr (tried .nii and .nii.gz)")

def find_prostate_mask(pid: str) -> str:
    # exact match first
    for ext in [".nii", ".nii.gz"]:
        p = os.path.join(ZONES_DIR, pid + ext)
        if os.path.exists(p):
            return p

    # fallback: old style "pid something.nii"
    for f in os.listdir(ZONES_DIR):
        if f.startswith(pid + " ") and (f.endswith(".nii") or f.endswith(".nii.gz")):
            return os.path.join(ZONES_DIR, f)

    raise FileNotFoundError(f"Prostate mask not found for {pid} in zonesTr")

# ------------ Build patient list -------------

# We build the patient list from underscore files pid_0001.nii / pid_0002.nii
patients = sorted({
    f.split("_")[0]
    for f in os.listdir(IMAGES_DIR)
    if f.endswith(".nii") and "_000" in f  # catches pid_0001.nii etc
})

print(f"Found {len(patients)} patients:", patients[:10])

if len(patients) == 0:
    print("❌ No usable cases found! Expected files like 000_0001.nii in imagesTr.")
    raise SystemExit

# ------------ Train / Test / Holdout Split --------------

train_ids, temp_ids = train_test_split(patients, test_size=0.30, random_state=42)
test_ids, holdout_ids = train_test_split(temp_ids, test_size=(1/3), random_state=42)

print(f"\nSplit summary:")
print(f"  Train:   {len(train_ids)} patients")
print(f"  Test:    {len(test_ids)} patients")
print(f"  Holdout: {len(holdout_ids)} patients\n")

# ------------ Process & Save ----------------

def process_and_save(pid: str, split: str):
    ch0_path, ch1_path, ch2_path = find_modalities(pid)

    # Load channel 0 first (needed for shape if we zero-fill missing channels)
    v0_xyz = zscore_nonzero(load_nifti(ch0_path))

    if ch1_path is None:
        v1_xyz = np.zeros_like(v0_xyz, dtype=np.float32)
    else:
        v1_xyz = zscore_nonzero(load_nifti(ch1_path))

    if ch2_path is None:
        v2_xyz = np.zeros_like(v0_xyz, dtype=np.float32)
    else:
        v2_xyz = zscore_nonzero(load_nifti(ch2_path))

    # Load lesion label + prostate mask (mostly for sanity checking alignment)
    lesion_path = find_lesion_label(pid)
    pros_path   = find_prostate_mask(pid)

    lesion_xyz = load_nifti(lesion_path)
    pros_xyz   = load_nifti(pros_path)

    # Sanity checks for shape match
    shp = v0_xyz.shape
    if v1_xyz.shape != shp or v2_xyz.shape != shp:
        raise ValueError(f"Shape mismatch among modalities for {pid}: {v0_xyz.shape}, {v1_xyz.shape}, {v2_xyz.shape}")
    if lesion_xyz.shape != shp:
        raise ValueError(f"Lesion label shape mismatch for {pid}: lesion {lesion_xyz.shape} vs image {shp}")
    if pros_xyz.shape != shp:
        raise ValueError(f"Prostate mask shape mismatch for {pid}: prostate {pros_xyz.shape} vs image {shp}")

    # Convert to Z,Y,X order for your patch-training code
    v0 = to_zyx(v0_xyz)
    v1 = to_zyx(v1_xyz)
    v2 = to_zyx(v2_xyz)

    lesion = to_zyx(lesion_xyz)
    # instance-coded 0..L -> binary
    lesion = (lesion > 0).astype(np.float32)

    img = np.stack([v0, v1, v2], axis=0).astype(np.float32)  # (3,Z,Y,X)

    # Index like your prostate script: count existing split files
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"image_{split}")]
    idx = len(existing)

    np.save(os.path.join(OUTPUT_DIR, f"image_{split}{idx:03d}.npy"), img)
    np.save(os.path.join(OUTPUT_DIR, f"label_{split}{idx:03d}.npy"), lesion)

    used = [os.path.basename(ch0_path)]
    used.append(os.path.basename(ch1_path) if ch1_path else "ZEROS")
    used.append(os.path.basename(ch2_path) if ch2_path else "ZEROS")

    print(f"✓ Saved {pid} → {split}{idx:03d}   channels: {used}")

if __name__ == "__main__":

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
