import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Point to ROI outputs
# -------------------------
DATA = "./data/lesion-roi-data"
PRED = "./result_lesion_roi"
OUT  = "./result_lesion_roi/png"
os.makedirs(OUT, exist_ok=True)

# Model input/output size used during training (must match train_lesion_unet.py)
FIXED_SIZE = (128, 128, 64)

# Change this to show a binary mask instead of soft probabilities
PRED_THRESHOLD = 0.2   # try 0.1 / 0.2 / 0.3
SHOW_BINARY_PRED = False  # True = thresholded mask, False = probability map


# -------------------------
# Make ROI volumes match model output shape
# -------------------------
def pad_or_center_crop_3d(x, target_shape=FIXED_SIZE):
    """
    Make x exactly target_shape by:
      - center-cropping if x is larger
      - zero-padding if x is smaller
    """
    th, tw, td = target_shape
    H, W, D = x.shape

    # center crop if larger
    hs = max(0, (H - th) // 2)
    ws = max(0, (W - tw) // 2)
    ds = max(0, (D - td) // 2)
    x_c = x[hs:hs + min(th, H), ws:ws + min(tw, W), ds:ds + min(td, D)]

    # pad if smaller
    out = np.zeros(target_shape, dtype=x.dtype)
    h, w, d = x_c.shape
    oh = (th - h) // 2
    ow = (tw - w) // 2
    od = (td - d) // 2
    out[oh:oh + h, ow:ow + w, od:od + d] = x_c
    return out


def load_test_image(idx):
    path = os.path.join(DATA, f"image_test{idx:03d}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing test image: {path}")
    img = np.load(path)
    return pad_or_center_crop_3d(img)

def load_test_label(idx):
    path = os.path.join(DATA, f"label_test{idx:03d}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing test label: {path}")
    lab = np.load(path)
    return pad_or_center_crop_3d(lab)

def extract_ids(filename):
    """
    Extract step and test index from filename:
    pred_stepXXXX_idYYY.npy
    """
    name = filename.replace(".npy", "")
    if "_id" in name:
        step = int(name.split("_")[1].replace("step", ""))
        pid  = int(name.split("_")[2].replace("id", ""))
        return step, pid
    else:
        step = int(name.split("_")[1].replace("step", ""))
        pid  = 0
        return step, pid


def plot_strip(img, gt, pred, savepath, title):
    """
    Make an 8-slice strip:
      Col 0: image
      Col 1: ground truth lesion
      Col 2: prediction (probability or binary mask)
    """
    nz = img.shape[2]
    stepz = max(1, nz // 8)
    slices = list(range(0, nz, stepz))[:8]

    fig, axarr = plt.subplots(len(slices), 3, figsize=(9, 16))
    fig.suptitle(title)

    for i, z in enumerate(slices):
        # Image
        axarr[i, 0].imshow(img[:, :, z], cmap="gray")
        axarr[i, 0].set_title(f"Image z={z}")
        axarr[i, 0].axis("off")

        # Ground truth (binary)
        axarr[i, 1].imshow(gt[:, :, z] > 0, cmap="gray", vmin=0, vmax=1)
        axarr[i, 1].set_title("GT lesion")
        axarr[i, 1].axis("off")

        # Prediction
        if SHOW_BINARY_PRED:
            axarr[i, 2].imshow(pred[:, :, z] > PRED_THRESHOLD, cmap="gray", vmin=0, vmax=1)
            axarr[i, 2].set_title(f"Pred > {PRED_THRESHOLD}")
        else:
            axarr[i, 2].imshow(pred[:, :, z], cmap="gray", vmin=0, vmax=1)
            axarr[i, 2].set_title("Prediction (prob)")
        axarr[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close()
    print(f"✓ Saved strip: {savepath}")


# -------------------------
# Run through predictions
# -------------------------
pred_files = sorted([f for f in os.listdir(PRED) if f.endswith(".npy") and f.startswith("pred_")])
print(f"Found {len(pred_files)} prediction files in {PRED}.")

for f in pred_files:
    print(f"Processing: {f}")
    step, pid = extract_ids(f)

    pred = np.load(os.path.join(PRED, f))
    img  = load_test_image(pid)
    gt   = load_test_label(pid)

    # safety check (now should match)
    if pred.shape != img.shape or gt.shape != img.shape:
        raise RuntimeError(f"Shape mismatch for id {pid:03d}: img={img.shape}, gt={gt.shape}, pred={pred.shape}")

    savepath = os.path.join(OUT, f"lesion_{pid:03d}_step{step:04d}.png")
    plot_strip(img, gt, pred, savepath, f"Lesion ROI | ID {pid} | step {step}")
