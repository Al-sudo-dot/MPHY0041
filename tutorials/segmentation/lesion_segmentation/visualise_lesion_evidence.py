import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG (override from terminal)
# ---------------------------
DATA_DIR = "./data/lesion-data"
PRED_DIR = os.environ.get("PRED_DIR", "./result/lesion_pw30")  # choose pw10/pw30/pw80
OUT_DIR  = os.environ.get("OUT_DIR", os.path.join(PRED_DIR, "png_evidence"))
os.makedirs(OUT_DIR, exist_ok=True)

BEST_ONLY = os.environ.get("BEST_ONLY", "1") == "1"  # default: only best/latest file
THRESH = float(os.environ.get("THRESH", "0.5"))      # threshold for Dice / binary mask
K_SLICES = int(os.environ.get("K", "8"))             # number of slices in the strip

# ---------------------------
# Helpers
# ---------------------------

def load_test_image(pid):
    """Returns T2 only as (Z,Y,X) float32."""
    path = os.path.join(DATA_DIR, f"image_test{pid:03d}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing test image: {path}")
    img = np.load(path)
    if img.ndim == 4:
        t2 = img[0]  # channel 0
    elif img.ndim == 3:
        t2 = img
    else:
        raise ValueError(f"Unexpected image shape {img.shape}")
    return t2.astype(np.float32)

def load_test_label(pid):
    """Returns GT lesion mask as (Z,Y,X) uint8 or None."""
    path = os.path.join(DATA_DIR, f"label_test{pid:03d}.npy")
    if not os.path.exists(path):
        return None
    lab = np.load(path)
    if lab.ndim == 4:
        lab = lab[0]
    return (lab > 0).astype(np.uint8)

def extract_step_and_id(fname):
    step = None
    pid = 0
    m_step = re.search(r"step(\d+)", fname)
    if m_step:
        step = int(m_step.group(1))
    m_id = re.search(r"id(\d+)", fname)
    if m_id:
        pid = int(m_id.group(1))
    return step, pid

def ensure_zyx(vol, ref_shape):
    """Try to match (Z,Y,X) to ref_shape using common permutations."""
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"Prediction must be 3D, got {vol.shape}")

    if vol.shape == ref_shape:
        return vol

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
    return vol  # fallback (may look rotated)

def dice_score(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return float((2 * inter + eps) / (denom + eps))

def choose_slices(gt_zyx, pred_prob_zyx, k):
    """
    If GT has lesion slices: pick those slices (spread to k).
    Else: pick top-k slices by predicted probability mass.
    """
    Z = pred_prob_zyx.shape[0]

    if gt_zyx is not None and gt_zyx.sum() > 0:
        per_slice = gt_zyx.reshape(Z, -1).sum(axis=1)
        z = np.where(per_slice > 0)[0]
        if len(z) > k:
            take = np.linspace(0, len(z) - 1, k).round().astype(int)
            z = z[take]
        return list(z)

    per_slice_p = pred_prob_zyx.reshape(Z, -1).sum(axis=1)
    if per_slice_p.sum() > 0:
        top = np.argsort(-per_slice_p)[:k]
        return sorted(list(top))

    step = max(1, Z // k)
    return list(range(0, Z, step))[:k]

def pick_pred_files(pred_dir, best_only=True):
    files = [f for f in os.listdir(pred_dir) if f.endswith(".npy") and "pred" in f]
    if not files:
        return []

    # only consider step-based preds if they exist
    step_files = [f for f in files if "step" in f]
    use = step_files if step_files else files

    if not best_only:
        return sorted(use)

    def step_num(f):
        m = re.search(r"step(\d+)", f)
        return int(m.group(1)) if m else -1

    best = max(use, key=step_num)
    return [best]

# ---------------------------
# Plotting
# ---------------------------

def plot_evidence_strip(t2, gt, pred_prob, pred_bin, z_slices, title, savepath):
    has_gt = gt is not None
    ncols = 3 if has_gt else 2

    fig, axarr = plt.subplots(len(z_slices), ncols, figsize=(4*ncols, 2*len(z_slices)))
    fig.suptitle(title)

    if len(z_slices) == 1:
        axarr = np.expand_dims(axarr, axis=0)

    for i, z in enumerate(z_slices):
        # T2
        axarr[i, 0].imshow(t2[z], cmap="gray")
        axarr[i, 0].set_title(f"T2 z={z}")
        axarr[i, 0].axis("off")

        if has_gt:
            axarr[i, 1].imshow(gt[z], cmap="gray")
            axarr[i, 1].set_title("GT lesion")
            axarr[i, 1].axis("off")

            # Probability heatmap (recommended)
            axarr[i, 2].imshow(pred_prob[z], cmap="hot", vmin=0, vmax=1)
            axarr[i, 2].set_title(f"Pred prob (>{THRESH} bin)")
            axarr[i, 2].axis("off")
        else:
            axarr[i, 1].imshow(pred_prob[z], cmap="hot", vmin=0, vmax=1)
            axarr[i, 1].set_title(f"Pred prob (>{THRESH} bin)")
            axarr[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close()
    print(f"✓ Saved evidence strip: {savepath}")

# ---------------------------
# MAIN
# ---------------------------

pred_files = pick_pred_files(PRED_DIR, best_only=BEST_ONLY)

print(f"PRED_DIR: {PRED_DIR}")
print(f"OUT_DIR:  {OUT_DIR}")
print(f"Found {len(pred_files)} prediction file(s). BEST_ONLY={BEST_ONLY}")

for fname in pred_files:
    step, pid = extract_step_and_id(fname)

    # Load data
    t2 = load_test_image(pid)   # (Z,Y,X)
    gt = load_test_label(pid)   # (Z,Y,X) or None

    # Load prediction (can be binary or probabilities)
    pred = np.load(os.path.join(PRED_DIR, fname))
    pred = ensure_zyx(pred, ref_shape=t2.shape)

    # If prediction looks binary already, convert to prob-like 0/1
    # Otherwise clip to [0,1] for heatmap display.
    if pred.dtype != np.float32 and pred.dtype != np.float64:
        pred_prob = pred.astype(np.float32)
    else:
        pred_prob = pred.astype(np.float32)

    pred_prob = np.clip(pred_prob, 0.0, 1.0)
    pred_bin = (pred_prob >= THRESH).astype(np.uint8)

    # Dice (if GT exists)
    if gt is not None and gt.sum() > 0:
        d = dice_score(pred_bin, gt)
        dice_txt = f"Dice={d:.3f}"
    else:
        dice_txt = "Dice=N/A"

    # choose slices
    z_slices = choose_slices(gt, pred_prob, K_SLICES)

    # Save path
    if step is None:
        outname = f"evidence_pred_id{pid:03d}.png"
        title = f"ID {pid} | {dice_txt} | THRESH={THRESH}"
    else:
        outname = f"evidence_pred_step{step:04d}_id{pid:03d}.png"
        title = f"ID {pid} step {step} | {dice_txt} | THRESH={THRESH}"

    savepath = os.path.join(OUT_DIR, outname)
    plot_evidence_strip(t2, gt, pred_prob, pred_bin, z_slices, title, savepath)
