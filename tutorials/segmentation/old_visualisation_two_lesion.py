import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Paths (edit if needed)
# -----------------------
DATA_DIR = "./data/lesion-data"
PRED_DIR = "./result_lesion"
OUT_DIR  = "./result_lesion/overlay_png"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Utilities
# -----------------------
def load_vol(split, idx, kind="image"):
    path = os.path.join(DATA_DIR, f"{kind}_{split}{idx:03d}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return np.load(path)

def load_pred_for_test_idx(test_idx, step=500):
    # Matches your training save name: pred_stepXXXX_idYYY.npy
    path = os.path.join(PRED_DIR, f"pred_step{step:04d}_id{test_idx:03d}.npy")
    if not os.path.exists(path):
        # fallback: pick latest prediction for that id if step not found
        cand = sorted([f for f in os.listdir(PRED_DIR)
                       if f.startswith("pred_step") and f.endswith(f"_id{test_idx:03d}.npy")])
        if not cand:
            raise FileNotFoundError(f"No prediction found for test id {test_idx:03d} in {PRED_DIR}")
        path = os.path.join(PRED_DIR, cand[-1])
        print("Using prediction:", os.path.basename(path))
    return np.load(path)

def pick_slice(img, gt=None, pred=None, mode="max_gt"):
    """
    mode:
      - "mid": middle slice
      - "max_gt": slice with most GT lesion (best for showing GT)
      - "max_pred": slice with most predicted lesion
    """
    nz = img.shape[2]
    if mode == "mid" or (gt is None and pred is None):
        return nz // 2

    if mode == "max_gt" and gt is not None:
        sums = gt.sum(axis=(0, 1))
        z = int(np.argmax(sums))
        return z if sums[z] > 0 else nz // 2

    if mode == "max_pred" and pred is not None:
        sums = pred.sum(axis=(0, 1))
        z = int(np.argmax(sums))
        return z if sums[z] > 0 else nz // 2

    return nz // 2

def overlay(ax, base, mask, alpha=0.45):
    ax.imshow(base, cmap="gray")
    ax.imshow(mask, alpha=alpha)   # uses default colormap (no hard-coded colors)
    ax.axis("off")

# -----------------------
# Main: make 3-panel figure
# -----------------------
def make_triptych(test_idx=0, step=500, thresh=0.5, slice_mode="max_gt", out_name=None):
    # Load volumes
    img = load_vol("test", test_idx, "image")
    gt  = load_vol("test", test_idx, "label")          # requires you saved label_testXXX.npy
    pred = load_pred_for_test_idx(test_idx, step=step)

    # Ensure shapes
    assert img.shape == gt.shape == pred.shape, (img.shape, gt.shape, pred.shape)

    # Pick slice
    z = pick_slice(img, gt=gt, pred=pred, mode=slice_mode)

    # Prepare masks for display
    gt_mask = (gt[:, :, z] > 0).astype(float)
    pr_mask = (pred[:, :, z] >= thresh).astype(float)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img[:, :, z], cmap="gray")
    axes[0].set_title("T2 Image")
    axes[0].axis("off")

    overlay(axes[1], img[:, :, z], gt_mask, alpha=0.50)
    axes[1].set_title("Ground Truth")

    overlay(axes[2], img[:, :, z], pr_mask, alpha=0.50)
    axes[2].set_title(f"Prediction (thr={thresh})")

    plt.tight_layout()

    if out_name is None:
        out_name = f"triptych_test{test_idx:03d}_step{step:04d}_z{z:02d}.png"
    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print("Saved:", out_path)

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    make_triptych(test_idx=0, step=500, thresh=0.5, slice_mode="max_gt")
