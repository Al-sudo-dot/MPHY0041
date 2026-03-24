import os, json, argparse
import numpy as np
import pandas as pd

# connected components
try:
    from scipy.ndimage import label as cc_label
except Exception as e:
    raise ImportError(
        "This script needs scipy for connected components. "
        "Install it (pip install scipy) or tell me and I'll provide a pure-python fallback."
    ) from e


def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred_bin.astype(bool)
    gt   = gt_bin.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2.0 * inter + eps) / (denom + eps))


def apply_roi_bbox(vol: np.ndarray, bbox):
    """
    bbox is [z0,z1,y0,y1,x0,x1] in end-exclusive indexing.
    """
    if isinstance(bbox, dict):
        z0, z1 = int(bbox["z0"]), int(bbox["z1"])
        y0, y1 = int(bbox["y0"]), int(bbox["y1"])
        x0, x1 = int(bbox["x0"]), int(bbox["x1"])
    else:
        z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox]

    out = np.zeros_like(vol)
    out[z0:z1, y0:y1, x0:x1] = vol[z0:z1, y0:y1, x0:x1]
    return out


def keep_largest_component(binmask: np.ndarray) -> np.ndarray:
    lab, n = cc_label(binmask)
    if n == 0:
        return binmask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return (lab == largest)


def remove_small_components(binmask: np.ndarray, min_vox: int) -> np.ndarray:
    if min_vox <= 1:
        return binmask
    lab, n = cc_label(binmask)
    if n == 0:
        return binmask
    counts = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[counts >= min_vox] = True
    keep[0] = False
    return keep[lab]


def load_gt(data_dir: str, idx: int) -> np.ndarray:
    """
    Try common GT naming patterns. Adjust if your dataset uses a different one.
    """
    candidates = [
        os.path.join(data_dir, f"label_test{idx:03d}.npy"),
        os.path.join(data_dir, f"label_holdout{idx:03d}.npy"),
        os.path.join(data_dir, f"label_{idx:03d}.npy"),
        os.path.join(data_dir, "labels", f"label_test{idx:03d}.npy"),
        os.path.join(data_dir, "labels", f"{idx:03d}.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            g = np.load(p)
            # allow (1,Z,Y,X) or (Z,Y,X)
            if g.ndim == 4:
                g = g[0]
            return (g > 0).astype(np.uint8)
    raise FileNotFoundError(f"Could not find GT label for id {idx:03d} in {data_dir}")


def load_pred(pred_dir: str, step: int, idx: int) -> np.ndarray:
    """
    Your preds are: pred_step8000_id###.npy
    """
    candidates = [
        os.path.join(pred_dir, f"pred_step{step}_id{idx:03d}.npy"),
        os.path.join(pred_dir, f"pred_step{step}_best_id{idx:03d}.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            pr = np.load(p).astype(np.float32)
            # allow (1,Z,Y,X) or (Z,Y,X)
            if pr.ndim == 4:
                pr = pr[0]
            return pr
    raise FileNotFoundError(f"Could not find prediction for id {idx:03d} in {pred_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--roi_json", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--n_cases", type=int, default=87)
    ap.add_argument("--min_vox", type=int, default=50)
    ap.add_argument("--keep_largest", action="store_true")
    ap.add_argument("--out_csv", default="summary_patch3d_sweep.csv")
    ap.add_argument("--thr_start", type=float, default=0.05)
    ap.add_argument("--thr_end", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.05)
    args = ap.parse_args()

    with open(args.roi_json, "r") as f:
        rois = json.load(f)

    # ✅ Your ROI keys are like: "image_test044.npy"
    def get_roi(i: int):
        key = f"image_test{i:03d}.npy"
        if key in rois:
            return rois[key]
        # helpful debug: show similar keys
        token = f"{i:03d}"
        near = [k for k in rois.keys() if token in k][:10]
        raise KeyError(f"No ROI found for case {i} with key {key}. Similar keys: {near}")

    thresholds = np.arange(args.thr_start, args.thr_end + 1e-9, args.thr_step)

    # preload
    preds = []
    gts = []
    rois_list = []
    for i in range(args.n_cases):
        preds.append(load_pred(args.pred_dir, args.step, i))
        gts.append(load_gt(args.data_dir, i))
        rois_list.append(get_roi(i))

    rows = []
    oracle_best = np.zeros(args.n_cases, dtype=np.float32)

    best_mean = -1.0
    best_thr = None
    best_per_case = None

    for thr in thresholds:
        dices = np.zeros(args.n_cases, dtype=np.float32)

        for i in range(args.n_cases):
            prob = preds[i]
            gt   = gts[i]

            # ROI mask
            prob_roi = apply_roi_bbox(prob, rois_list[i])
            gt_roi   = apply_roi_bbox(gt,   rois_list[i])

            # threshold
            pred_bin = (prob_roi >= thr)

            # postproc
            pred_bin = remove_small_components(pred_bin, args.min_vox)
            if args.keep_largest:
                pred_bin = keep_largest_component(pred_bin)

            d = dice_score(pred_bin, gt_roi)
            dices[i] = d
            if d > oracle_best[i]:
                oracle_best[i] = d

        mean_d = float(dices.mean())
        med_d  = float(np.median(dices))

        rows.append({"thr": float(thr), "mean_dice": mean_d, "median_dice": med_d})
        print(f"thr={thr:.2f} mean={mean_d:.4f} median={med_d:.4f}")

        if mean_d > best_mean:
            best_mean = mean_d
            best_thr = float(thr)
            best_per_case = dices.copy()

    oracle_mean = float(oracle_best.mean())
    oracle_median = float(np.median(oracle_best))

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    per_case_csv = args.out_csv.replace(".csv", f"_bestthr_{best_thr:.2f}_percase.csv")
    pd.DataFrame({"id": np.arange(args.n_cases), "dice_bestthr": best_per_case}).to_csv(per_case_csv, index=False)

    oracle_csv = args.out_csv.replace(".csv", "_oracle_percase.csv")
    pd.DataFrame({"id": np.arange(args.n_cases), "dice_oracle": oracle_best}).to_csv(oracle_csv, index=False)

    print("\n=== BEST GLOBAL THRESHOLD ===")
    print(f"best_thr = {best_thr:.2f}")
    print(f"best_mean_dice = {best_mean:.4f}")
    print(f"oracle_mean_dice = {oracle_mean:.4f}")
    print(f"oracle_median_dice = {oracle_median:.4f}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {per_case_csv}")
    print(f"wrote: {oracle_csv}")


if __name__ == "__main__":
    main()
