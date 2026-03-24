import argparse, numpy as np
from postprocess_utils import postprocess_lesion, dice

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pred .npy (probabilities)")
    ap.add_argument("--gt", required=True, help="gt .npy (0/1)")
    ap.add_argument("--min_vox", type=int, default=50)
    ap.add_argument("--keep_largest", action="store_true")
    args = ap.parse_args()

    prob = np.load(args.pred)
    gt   = (np.load(args.gt) > 0).astype(np.uint8)

    print("thr\tpred_vox\tdice_raw\tpred_vox_pp\tdice_pp")
    for thr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        raw = (prob >= thr).astype(np.uint8)
        d_raw = dice(raw, gt)
        pp = postprocess_lesion(prob, thr=thr, min_vox=args.min_vox, keep_largest=args.keep_largest)
        d_pp = dice(pp, gt)
        print(f"{thr:.2f}\t{raw.sum()}\t{d_raw:.4f}\t{pp.sum()}\t{d_pp:.4f}")

if __name__ == "__main__":
    main()
