import os, argparse, csv
import numpy as np

from postprocess_utils import postprocess_mask  # must exist from your earlier step

def dice(a, b, eps=1e-8):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return (2.0*inter + eps) / (a.sum() + b.sum() + eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--ids", nargs="*", type=int, default=None,
                    help="Optional list of test IDs (e.g. 0 1 2). If omitted, infer from gt_dir.")
    ap.add_argument("--thr_list", type=float, nargs="*", default=[0.5,0.6,0.7,0.8,0.85,0.9])
    ap.add_argument("--min_vox", type=int, default=50)
    ap.add_argument("--keep_largest", action="store_true")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # infer IDs from GT filenames if not given
    if args.ids is None or len(args.ids) == 0:
        ids = []
        for fn in os.listdir(args.gt_dir):
            if fn.startswith("label_test") and fn.endswith(".npy"):
                s = fn[len("label_test"):-len(".npy")]
                if s.isdigit():
                    ids.append(int(s))
        ids = sorted(ids)
    else:
        ids = args.ids

    rows = []
    for tid in ids:
        gt_path = os.path.join(args.gt_dir, f"label_test{tid:03d}.npy")
        pred_path = os.path.join(args.pred_dir, f"pred_step1200_id{tid:03d}.npy")
        if (not os.path.exists(gt_path)) or (not os.path.exists(pred_path)):
            continue

        gt = np.load(gt_path) > 0
        prob = np.load(pred_path)

        for thr in args.thr_list:
            raw = prob >= thr
            raw_d = dice(raw, gt)
            pp = postprocess_mask(raw, min_vox=args.min_vox, keep_largest=args.keep_largest)
            pp_d = dice(pp, gt)

            rows.append({
                "id": tid,
                "thr": thr,
                "gt_vox": int(gt.sum()),
                "pred_vox_raw": int(raw.sum()),
                "dice_raw": float(raw_d),
                "pred_vox_pp": int(pp.sum()),
                "dice_pp": float(pp_d),
            })

    # write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["id","thr","gt_vox","pred_vox_raw","dice_raw","pred_vox_pp","dice_pp"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] wrote {len(rows)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
