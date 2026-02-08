import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ---- CHANGE THIS ----
PATIENT_ID = "067"
DATASET_ROOT = "/Users/anastasia/Desktop/prostate cancer/Dataset_prostate.nosync/imagesTr"
OUT_DIR = f"./patient_{PATIENT_ID}_png"
os.makedirs(OUT_DIR, exist_ok=True)

# Files we want
files = [
    f"{PATIENT_ID}.nii",
    f"{PATIENT_ID}_0001.nii",
    f"{PATIENT_ID}_0002.nii",
]

def save_volume_as_png(path, tag):
    vol = nib.load(path).get_fdata()

    # Normalize for visualization
    vmin, vmax = vol.min(), vol.max()
    vol = (vol - vmin) / (vmax - vmin + 1e-8)

    # Pick slices
    z_slices = [0, vol.shape[2]//4, vol.shape[2]//2, 3*vol.shape[2]//4, vol.shape[2]-1]

    for i, z in enumerate(z_slices):
        plt.figure(figsize=(4,4))
        plt.imshow(vol[:,:,z], cmap="gray")
        plt.axis("off")
        plt.title(f"{tag} z={z}")
        png_path = os.path.join(OUT_DIR, f"{tag}_z{z}.png")
        plt.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close()
        print("✔ Saved", png_path)


if __name__ == "__main__":
    for f in files:
        full_path = os.path.join(DATASET_ROOT, f)
        if os.path.exists(full_path):
            print(f"\nProcessing {full_path}")
            save_volume_as_png(full_path, f.replace(".nii",""))
        else:
            print(f"⚠ File not found: {full_path}")

    print("\n🎉 DONE! Check folder:", OUT_DIR)
