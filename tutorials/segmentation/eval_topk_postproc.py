import argparse
import os
import numpy as np
import pandas as pd
from scipy.ndimage import label


def dice(a, b, eps=1e-8):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    return (2 * inter + eps) / (a.sum() + b.sum() + eps)


def keep_topk(mask, k):
    lab, n = label(mask)

    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    sizes = []
    for i in range(1, n + 1):
        sizes.append(((lab == i).sum(), i))

    sizes.sort(reverse=True)
    keep = [i for _, i in sizes[:k]]

    out = np.zeros_like(mask, dtype=np.uint8)
    for i in keep:
        out[lab == i] = 1

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--thr", type=float, default=0.29)
    parser.add_argument("--min_vox", type=int, default=100)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--out_csv", type=str, default=None)

    args = parser.parse_args()

    dices = []
    rows = []

    for pid in range(87):
        pred_path = f"{args.pred_dir}/pred_step8000_id{pid:03d}.npy"
        gt_path = f"{args.data_dir}/label_test{pid:03d}.npy"

        if not os.path.exists(pred_path):
            print(f"[skip] missing pred: {pred_path}")
            continue

        if not os.path.exists(gt_path):
            print(f"[skip] missing gt: {gt_path}")
            continue

        prob = np.load(pred_path)
        gt = (np.load(gt_path) > 0).astype(np.uint8)

        pred = (prob > args.thr).astype(np.uint8)
        pred = keep_topk(pred, args.topk)

        if pred.sum() < args.min_vox:
            pred[:] = 0

        d = dice(pred, gt)
        dices.append(d)

        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())

        rows.append({
            "id": pid,
            "dice": float(d),
            "pred_vox": int(pred.sum()),
            "gt_vox": int(gt.sum()),
            "fp_vox": fp,
            "fn_vox": fn,
            "topk": int(args.topk),
            "thr": float(args.thr),
            "min_vox": int(args.min_vox),
        })

    if len(dices) == 0:
        print("No cases evaluated.")
        return

    mean_d = float(np.mean(dices))
    median_d = float(np.median(dices))

    print(f"Mean dice: {mean_d}")
    print(f"Median dice: {median_d}")

    if args.out_csv is not None:
        df = pd.DataFrame(rows).sort_values("dice")
        df.to_csv(args.out_csv, index=False)
        print("saved csv:", args.out_csv)


if __name__ == "__main__":
    main()
