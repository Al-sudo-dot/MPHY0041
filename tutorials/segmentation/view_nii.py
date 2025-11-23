import os
import sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Usage:
#   python view_nii.py /path/to/file.nii.gz
# ------------------------------

if len(sys.argv) < 2:
    print("Usage: python view_nii.py <path_to_nii>")
    sys.exit(1)

nii_path = sys.argv[1]

if not os.path.exists(nii_path):
    print(f"❌ File not found: {nii_path}")
    sys.exit(1)

print(f"Loading NIfTI: {nii_path}")
img = nib.load(nii_path).get_fdata()

# pick middle slice of axial plane
mid_slice = img[:, :, img.shape[2] // 2]

# create output directory
output_dir = "./nii_png"
os.makedirs(output_dir, exist_ok=True)

# output file name
base = os.path.basename(nii_path).replace(".nii.gz", "").replace(".nii", "")
png_path = os.path.join(output_dir, base + ".png")

# save PNG
plt.imshow(mid_slice, cmap="gray")
plt.axis("off")
plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=200)
plt.close()

print(f"✅ Saved PNG to: {png_path}")
