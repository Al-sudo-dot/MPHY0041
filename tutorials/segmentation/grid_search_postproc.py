import os
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.ndimage import label


def dice(a, b, eps=1e-8):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    return (2.0 * inter + eps) / (a.sum() + b.sum() + eps)


def keep_topk(mask, k):
    lab, n = label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    comps = []
    for i in range(1, n + 1):
        sz = int((lab == i).sum())
        comps.append((sz, i))

    comps.sort(reverse=True)
    keep_ids = [i for _, i in comps[:k]]

    out = np.zeros_like(mask, dtype=np.uint8)
    for i in keep_ids:
        out[lab == i] = 1
    return out


def postprocess(prob, thr, min_vox, topk):
    pred = (prob >= thr).astype(np.uint8)

    if topk is not None and topk > 0:
        pred = keep_topk(pred, topk)

    if pred.sum() < min_vox:
        pred[:] = 0

    return pred


def load_case_paths(data_dir, pred_dir, n_cases=87, step=8000):
    cases = []
    for pid in range(n_cases):
        gt_path = os.path.join(data_dir, f"label_test{pid:03d}.npy")
        pred_path = os.path.join(pred_dir, f"pred_step{step}_id{pid:03d}.npy")

        if not os.path.exists(gt_path):
            print(f"[skip] missing gt: {gt_path}")
            continue
        if not os.path.exists(pred_path):
            print(f"[skip] missing pred: {pred_path}")
            continue

        cases.append((pid, gt_path, pred_path))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--step", type=int, default=8000)
    ap.add_argument("--n_cases", type=int, default=87)

    ap.add_argument("--thr_list", nargs="+", type=float, required=True)
    ap.add_argument("--min_vox_list", nargs="+", type=int, required=True)
    ap.add_argument("--topk_list", nargs="+", type=int, required=True)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--best_percase_csv", default=None)

    args = ap.parse_args()

    cases = load_case_paths(
        data_dir=args.data_dir,
        pred_dir=args.pred_dir,
        n_cases=args.n_cases,
        step=args.step,
    )

    if len(cases) == 0:
        print("No valid cases found.")
        return

    # preload everything once
    gt_dict = {}
    prob_dict = {}
    for pid, gt_path, pred_path in cases:
        gt_dict[pid] = (np.load(gt_path) > 0).astype(np.uint8)
        prob_dict[pid] = np.load(pred_path)

    rows = []
    best_score = -1.0
    best_cfg = None
    best_percase = None

    for thr, min_vox, topk in itertools.product(
        args.thr_list, args.min_vox_list, args.topk_list
    ):
        dices = []
        percase_rows = []

        for pid, _, _ in cases:
            gt = gt_dict[pid]
            prob = prob_dict[pid]

            pred = postprocess(prob, thr=thr, min_vox=min_vox, topk=topk)
            d = dice(pred, gt)

            dices.append(d)

            percase_rows.append({
                "id": pid,
                "dice": float(d),
                "pred_vox": int(pred.sum()),
                "gt_vox": int(gt.sum()),
                "thr": float(thr),
                "min_vox": int(min_vox),
                "topk": int(topk),
            })

        mean_d = float(np.mean(dices))
        median_d = float(np.median(dices))

        rows.append({
            "thr": float(thr),
            "min_vox": int(min_vox),
            "topk": int(topk),
            "mean_dice": mean_d,
            "median_dice": median_d,
        })

        print(
            f"thr={thr:.3f} min_vox={min_vox} topk={topk} "
            f"mean={mean_d:.4f} median={median_d:.4f}"
        )

        if mean_d > best_score:
            best_score = mean_d
            best_cfg = (thr, min_vox, topk)
            best_percase = percase_rows

    df = pd.DataFrame(rows).sort_values(
        ["mean_dice", "median_dice"], ascending=False
    )
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved summary: {args.out_csv}")

    if best_cfg is not None:
        thr, min_vox, topk = best_cfg
        print("\n=== BEST CONFIG ===")
        print(f"thr = {thr}")
        print(f"min_vox = {min_vox}")
        print(f"topk = {topk}")
        print(f"best_mean_dice = {best_score:.4f}")

        if args.best_percase_csv is not None and best_percase is not None:
            pd.DataFrame(best_percase).sort_values("dice").to_csv(
                args.best_percase_csv, index=False
            )
            print(f"Saved best per-case CSV: {args.best_percase_csv}")


if __name__ == "__main__":
    main()
