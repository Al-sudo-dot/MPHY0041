import argparse, numpy as np

def dice(a,b,eps=1e-6):
    inter = (a & b).sum()
    return (2*inter + eps) / (a.sum() + b.sum() + eps)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--lesion_value", type=int, default=1)
    ap.add_argument("--thr_list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args=ap.parse_args()

    gt_raw=np.load(args.lab)
    pr=np.load(args.pred).astype(np.float32)
    if pr.ndim==4: pr=pr[0]
    gt=(gt_raw==args.lesion_value)

    thrs=[float(x) for x in args.thr_list.split(",")]
    best = (-1, None, None)

    print("thr\tpred_vox\tgt_vox\tinter\tDice")
    for t in thrs:
        pb = (pr>=t)
        inter = (pb & gt).sum()
        d = (2*inter + 1e-6) / (pb.sum() + gt.sum() + 1e-6)
        print(f"{t:.2f}\t{pb.sum()}\t{gt.sum()}\t{inter}\t{d:.4f}")
        if d > best[0]:
            best = (d, t, pb.sum())

    print(f"\nBEST: Dice={best[0]:.4f} at thr={best[1]:.2f} (pred_vox={best[2]})")

if __name__=="__main__":
    main()
