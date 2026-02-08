import os
import re
import numpy as np
import matplotlib.pyplot as plt

DATA = "./data/lesion-data"

# Default folder, but you can override from terminal
PRED = os.environ.get("PRED_DIR", "./result/lesion")
OUT  = os.environ.get("OUT_DIR", os.path.join(PRED, "png"))

os.makedirs(OUT, exist_ok=True)
BEST_ONLY = os.environ.get("BEST_ONLY", "0") == "1"


# ---------------------------
# Loading helpers
# ---------------------------

def load_test_image(idx):
    """
    Loads the multi-modal test image saved by prepare_lesion_data.py.
    Expected: (3, Z, Y, X)
    We display T2 = channel 0.
    Returns: (Z, Y, X)
    """
    path = os.path.join(DATA, f"image_test{idx:03d}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing test image: {path}")
    img = np.load(path)

    if img.ndim == 3:
        t2 = img
    elif img.ndim == 4:
        t2 = img[0]  # channel 0 = T2
    else:
        raise ValueError(f"Unexpected image shape {img.shape} in {path}")

    return t2.astype(np.float32)

def load_test_label(idx):
    """
    Loads GT lesion label saved by prepare_lesion_data.py.
    Expected: (Z,Y,X) or (1,Z,Y,X).
    Returns: (Z,Y,X) binary or None if missing.
    """
    path = os.path.join(DATA, f"label_test{idx:03d}.npy")
    if not os.path.exists(path):
        return None
    lab = np.load(path)
    if lab.ndim == 4:
        lab = lab[0]
    return (lab > 0).astype(np.uint8)

def ensure_zyx(vol, ref_shape=None):
    """
    Ensure a volume is (Z,Y,X). If ref_shape given, try permutations to match it.
    """
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"Prediction must be 3D, got shape {vol.shape}")

    if ref_shape is not None and vol.shape != ref_shape:
        candidates = [
            vol,
            np.transpose(vol, (2, 0, 1)),
            np.transpose(vol, (1, 0, 2)),
            np.transpose(vol, (0, 2, 1)),
            np.transpose(vol, (2, 1, 0)),
            np.transpose(vol, (1, 2, 0)),
        ]
        for c in candidates:
            if c.shape == ref_shape:
                return c
    return vol

# ---------------------------
# Dice
# ---------------------------

def dice_score(pred_bin, gt_bin, eps=1e-6):
    """
    pred_bin, gt_bin: (Z,Y,X) 0/1
    """
    inter = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return float((2 * inter + eps) / (denom + eps))

# ---------------------------
# Filename parsing
# ---------------------------

def extract_ids(filename):
    """
    Accepts: pred_step0050_id000.npy etc.
    Returns: (step, id)
    """
    name = filename.replace(".npy", "")
    step = None
    pid = None

    m_step = re.search(r"step(\d+)", name)
    if m_step:
        step = int(m_step.group(1))

    m_id = re.search(r"id(\d+)", name)
    if m_id:
        pid = int(m_id.group(1))
    else:
        pid = 0
    return step, pid

# ---------------------------
# Lesion-centric slice selection
# ---------------------------

def choose_slices(img_zyx, gt_zyx, pred_zyx, k=8):
    """
    Return a list of slice indices (z) to plot.
    Priority:
      1) If GT has lesion slices -> pick those (up to k, spread out if many)
      2) Else pick slices where prediction area is largest
      3) Else fallback evenly spaced
    """
    Z = img_zyx.shape[0]

    if gt_zyx is not None and gt_zyx.sum() > 0:
        per_slice = gt_zyx.reshape(Z, -1).sum(axis=1)
        z_idxs = np.where(per_slice > 0)[0]

        # If too many slices, spread them out
        if len(z_idxs) > k:
            take = np.linspace(0, len(z_idxs) - 1, k).round().astype(int)
            z_idxs = z_idxs[take]
        return list(z_idxs)

    # No GT lesion visible → use prediction “strongest slices”
    per_slice_p = pred_zyx.reshape(Z, -1).sum(axis=1)
    if per_slice_p.sum() > 0:
        # take top-k by predicted area, then sort for nice viewing order
        top = np.argsort(-per_slice_p)[:k]
        return sorted(list(top))

    # fallback evenly spaced
    step = max(1, Z // k)
    return list(range(0, Z, step))[:k]

# ---------------------------
# Plotting
# ---------------------------

def plot_strip(img_zyx, gt_zyx, pred_zyx, savepath, title, z_slices):
    has_gt = gt_zyx is not None
    ncols = 3 if has_gt else 2

    fig, axarr = plt.subplots(len(z_slices), ncols, figsize=(4*ncols, 2*len(z_slices)))
    fig.suptitle(title)

    if len(z_slices) == 1:
        axarr = np.expand_dims(axarr, axis=0)

    for i, z in enumerate(z_slices):
        axarr[i, 0].imshow(img_zyx[z], cmap="gray")
        axarr[i, 0].set_title(f"T2 z={z}")
        axarr[i, 0].axis("off")

        if has_gt:
            axarr[i, 1].imshow(gt_zyx[z], cmap="gray")
            axarr[i, 1].set_title("GT lesion")
            axarr[i, 1].axis("off")

            axarr[i, 2].imshow(pred_zyx[z], cmap="gray")
            axarr[i, 2].set_title("Prediction")
            axarr[i, 2].axis("off")
        else:
            axarr[i, 1].imshow(pred_zyx[z], cmap="gray")
            axarr[i, 1].set_title("Prediction")
            axarr[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(savepath, dpi=120)
    plt.close()
    print(f"✓ Saved strip: {savepath}")

# ---------------------------
# MAIN
# ---------------------------

all_preds = sorted([f for f in os.listdir(PRED) if f.endswith(".npy")])

if BEST_ONLY:
    # choose the latest step file (highest step number)
    def step_num(f):
        m = re.search(r"step(\d+)", f)
        return int(m.group(1)) if m else -1

    best_file = max(all_preds, key=step_num)
    pred_files = [best_file]
    print(f"BEST_ONLY enabled → visualising {best_file}")
else:
    pred_files = all_preds


print(f"Found {len(pred_files)} prediction files in {PRED}.")

for f in pred_files:
    step, pid = extract_ids(f)

    pred = np.load(os.path.join(PRED, f))
    img = load_test_image(pid)
    gt  = load_test_label(pid)

    pred = ensure_zyx(pred, ref_shape=img.shape)

    # if pred is probs, threshold; if already 0/1, this is fine
    pred_bin = (pred > 0.5).astype(np.uint8)

    # choose lesion-centric slices
    z_slices = choose_slices(img, gt, pred_bin, k=8)

    # dice (only if GT exists)
    if gt is not None and gt.sum() > 0:
        d = dice_score(pred_bin, gt)
        dice_txt = f"Dice={d:.3f}"
    else:
        dice_txt = "Dice=N/A"

    if step is None:
        savepath = os.path.join(OUT, f"pred_{pid:03d}.png")
        title = f"ID {pid} | {dice_txt}"
    else:
        savepath = os.path.join(OUT, f"pred_{pid:03d}_step{step:04d}.png")
        title = f"ID {pid} step {step} | {dice_txt}"

    plot_strip(img, gt, pred_bin, savepath, title, z_slices)
