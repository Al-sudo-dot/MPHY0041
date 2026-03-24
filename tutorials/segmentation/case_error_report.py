import os
import argparse
import numpy as np
import pandas as pd


def dice(a, b, eps=1e-8):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    return (2 * inter + eps) / (a.sum() + b.sum() + eps)


def postprocess(pred, min_vox=100, keep_largest=True):

    if pred.sum() == 0:
        return pred

    if keep_largest:
        import scipy.ndimage as ndi

        labels, num = ndi.label(pred)
        if num > 0:
            sizes = ndi.sum(pred, labels, range(1, num + 1))
            largest = np.argmax(sizes) + 1
            pred = (labels == largest)

    if min_vox > 0:
        if pred.sum() < min_vox:
            pred[:] = 0

    return pred.astype(np.uint8)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--thr", type=float, required=True)
    parser.add_argument("--min_vox", type=int, default=100)
    parser.add_argument("--keep_largest", action="store_true")
    parser.add_argument("--out_csv", required=True)

    args = parser.parse_args()

    rows = []

    pred_files = sorted(os.listdir(args.pred_dir))

    for f in pred_files:

        if not f.endswith(".npy"):
            continue

        pid = int(f.split("_id")[-1].split(".")[0])

        prob = np.load(os.path.join(args.pred_dir, f))

        pred = (prob >= args.thr).astype(np.uint8)

        pred = postprocess(pred, args.min_vox, args.keep_largest)

        gt = np.load(os.path.join(args.data_dir, f"label_test{pid:03d}.npy"))

        d = dice(pred, gt)

        fp = ((pred == 1) & (gt == 0)).sum()
        fn = ((pred == 0) & (gt == 1)).sum()

        rows.append({
            "id": pid,
            "dice": d,
            "pred_vox": int(pred.sum()),
            "gt_vox": int(gt.sum()),
            "fp_vox": int(fp),
            "fn_vox": int(fn),
        })

    df = pd.DataFrame(rows).sort_values("dice")

    df.to_csv(args.out_csv, index=False)

    print("Saved:", args.out_csv)

    print("\nWorst 10 cases:")
    print(df.head(10))

    print("\nBest 10 cases:")
    print(df.tail(10))


if __name__ == "__main__":
    main()
