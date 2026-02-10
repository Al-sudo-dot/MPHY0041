import os, numpy as np, nibabel as nib

DATASET_ROOT = "/cs/student/projects4/misc/alukic/datasets/Dataset_prostate.nosync"
IMAGES_DIR   = os.path.join(DATASET_ROOT, "imagesTr")
LABELS_DIR   = os.path.join(DATASET_ROOT, "labelsTr")

def load(path): return nib.load(path).get_fdata()

pids = sorted([f.replace(".nii","") for f in os.listdir(IMAGES_DIR) if f.endswith(".nii") and "_" not in f])[:5]

for pid in pids:
    m = load(os.path.join(LABELS_DIR, f"{pid}.nii"))
    vals, cnts = np.unique(m, return_counts=True)
    print(pid, "unique labels:", list(zip(vals.tolist(), cnts.tolist()))[:10], " ...")
    print("mask sum:", float(m.sum()), "nonzero voxels:", int((m>0).sum()))
    print("-"*60)
