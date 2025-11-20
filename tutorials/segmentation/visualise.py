import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "./data/prostate-data"
RESULT_DIR = "./result"
OUT_DIR = "./result/png"

os.makedirs(OUT_DIR, exist_ok=True)

# Load prediction files
pred_files = sorted([f for f in os.listdir(RESULT_DIR) if f.startswith("pred_") and f.endswith(".npy")])

if len(pred_files) == 0:
    print("❌ No prediction files found.")
    exit()

def load_test_image(pred_name):
    """
    Convert pred_023.npy → image_test023.npy
    """
    pid = pred_name.replace("pred_", "").replace(".npy", "")
    img_file = f"image_test{pid}.npy"
    path = os.path.join(DATA_DIR, img_file)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing test image: {path}")

    return np.load(path)


for f in pred_files:

    pred = np.load(os.path.join(RESULT_DIR, f))   # (160,160,64)
    img  = load_test_image(f)                     # (160,160,64)

    # Select 8 evenly-spaced slices
    num_slices = 8
    depth = pred.shape[2]
    indices = np.linspace(0, depth-1, num_slices).astype(int)

    fig, axes = plt.subplots(num_slices, 2, figsize=(6, num_slices*2))

    for i, z in enumerate(indices):

        axes[i,0].imshow(img[:,:,z], cmap="gray")
        axes[i,0].set_title(f"Image z={z}")
        axes[i,0].axis("off")

        axes[i,1].imshow(pred[:,:,z], cmap="gray")
        axes[i,1].set_title("Prediction")
        axes[i,1].axis("off")

    out_path = os.path.join(OUT_DIR, f.replace(".npy", ".png"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"✓ Saved strip: {out_path}")

print("\n✔ All image strips saved in ./result/png/")
