import os
import re
import numpy as np

DATA = "./data/lesion-data"

def load_gt(pid):
    p = os.path.join(DATA, f"label_test{pid:03d}.npy")
    if not os.path.exists(p):
        return None
    gt = np.load(p)
    if gt.ndim == 4:
        gt = gt[0]
    gt = (gt > 0).astype(np.uint8)
    return gt

def dice(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2*inter + eps) / (denom + eps))

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

def best_step_dice(result_dir):
    preds = [f for f in os.listdir(result_dir) if f.endswith(".npy") and "pred_step" in f]
    if not preds:
        return None

    best = (-1.0, None)  # (dice, filename)
    for f in preds:
        step, pid = extract_step_and_id(f)
        gt = load_gt(pid)
        if gt is None or gt.sum() == 0:
            continue
        pred = np.load(os.path.join(result_dir, f)).astype(np.uint8)
        d = dice(pred, gt)
        if d > best[0]:
            best = (d, f)

    return best

def main():
    dirs = [d for d in os.listdir("./result") if d.startswith("lesion_pw")]
    dirs = sorted(dirs)

    if not dirs:
        print("No result dirs found. Expected folders like ./result/lesion_pw30")
        return

    print("POS_WEIGHT experiment results (best Dice over saved steps)")
    print("--------------------------------------------------------")
    for d in dirs:
        full = os.path.join("./result", d)
        out = best_step_dice(full)
        if out is None or out[1] is None:
            print(f"{d:15s}  Dice=N/A  (no usable GT or no preds)")
        else:
            best_d, best_file = out
            print(f"{d:15s}  Dice={best_d:.3f}  best={best_file}")

if __name__ == "__main__":
    main()
