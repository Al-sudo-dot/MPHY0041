import argparse
import numpy as np

def dice(bin_pred, bin_gt, eps=1e-8):
    bin_pred = bin_pred.astype(bool)
    bin_gt   = bin_gt.astype(bool)
    inter = np.logical_and(bin_pred, bin_gt).sum()
    a = bin_pred.sum()
    b = bin_gt.sum()
    return (2.0 * inter + eps) / (a + b + eps), int(a), int(b), int(inter)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True, help="GT label .npy (Z,Y,X)")
    ap.add_argument("--pred", required=True, help="pred prob .npy (Z,Y,X) or (1,Z,Y,X)")
    ap.add_argument("--lesion_value", type=int, default=1, help="GT lesion label value (default 1)")
    ap.add_argument("--thr_list", default="0.8,0.85,0.9", help="comma-separated thresholds")
    args = ap.parse_args()

    gt = np.load(args.lab)
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt = gt[0]
    if gt.ndim != 3:
        raise ValueError(f"GT must be (Z,Y,X). Got {gt.shape}")

    pr = np.load(args.pred).astype(np.float32)
    if pr.ndim == 4 and pr.shape[0] == 1:
        pr = pr[0]
    if pr.ndim != 3:
        raise ValueError(f"Pred must be (Z,Y,X) or (1,Z,Y,X). Got {pr.shape}")

    gt_bin = (gt == int(args.lesion_value)).astype(np.uint8)

    thrs = [float(x) for x in args.thr_list.split(",") if x.strip()]

    print("thr\tpred_vox\tgt_vox\tinter\tDice")
    for t in thrs:
        pr_bin = (pr >= t).astype(np.uint8)
        d, pv, gv, inter = dice(pr_bin, gt_bin)
        print(f"{t:.2f}\t{pv}\t{gv}\t{inter}\t{d:.4f}")

if __name__ == "__main__":
    main()
