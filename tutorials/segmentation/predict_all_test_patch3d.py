import os, json, argparse
import numpy as np
import torch
from model_unet3d_small import UNet3DSmall


def clamp(a, lo, hi):
    return max(lo, min(hi, a))


def load_roi(rois, split_prefix, idx):
    # try common key styles
    keys = [f"{idx:03d}", str(idx), f"{split_prefix}{idx:03d}"]
    for k in keys:
        if k in rois:
            r = rois[k]
            if isinstance(r, dict):
                return int(r["z0"]), int(r["z1"]), int(r["y0"]), int(r["y1"]), int(r["x0"]), int(r["x1"])
            return tuple(int(v) for v in r)
    return None


@torch.no_grad()
def sliding_window_predict(img_czyx, model, roi, pz, py, px, stride_z=16, stride_y=48, stride_x=48, device="cuda"):
    C, Z, Y, X = img_czyx.shape
    z0, z1, y0, y1, x0, x1 = roi

    # clamp ROI bounds
    z0, z1 = clamp(z0, 0, Z), clamp(z1, 0, Z)
    y0, y1 = clamp(y0, 0, Y), clamp(y1, 0, Y)
    x0, x1 = clamp(x0, 0, X), clamp(x1, 0, X)

    # output accumulators
    prob_sum = np.zeros((Z, Y, X), dtype=np.float32)
    w_sum    = np.zeros((Z, Y, X), dtype=np.float32)

    hz, hy, hx = pz // 2, py // 2, px // 2

    # choose centers within ROI
    cz_list = list(range(z0 + hz, max(z0 + hz + 1, z1 - hz + 1), stride_z))
    cy_list = list(range(y0 + hy, max(y0 + hy + 1, y1 - hy + 1), stride_y))
    cx_list = list(range(x0 + hx, max(x0 + hx + 1, x1 - hx + 1), stride_x))

    # ensure last coverage near edge
    if cz_list and cz_list[-1] != z1 - hz:
        cz_list.append(z1 - hz)
    if cy_list and cy_list[-1] != y1 - hy:
        cy_list.append(y1 - hy)
    if cx_list and cx_list[-1] != x1 - hx:
        cx_list.append(x1 - hx)

    model.eval()

    for cz in cz_list:
        for cy in cy_list:
            for cx in cx_list:
                zA, zB = cz - hz, cz + hz
                yA, yB = cy - hy, cy + hy
                xA, xB = cx - hx, cx + hx

                patch = img_czyx[:, zA:zB, yA:yB, xA:xB]
                if patch.shape[1:] != (pz, py, px):
                    continue

                x = torch.from_numpy(patch[None]).to(device)  # (1,C,pz,py,px)
                logits = model(x)[0, 0].detach().cpu().numpy()

                prob_sum[zA:zB, yA:yB, xA:xB] += logits
                w_sum[zA:zB, yA:yB, xA:xB] += 1.0
    
    w_sum[w_sum == 0] = 1.0
    logit_avg = prob_sum / w_sum
    out = 1 / (1 + np.exp(-logit_avg))   # sigmoid applied once
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pth or ckpt_stepXXXXX.pth")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--roi_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=87)
    ap.add_argument("--pz", type=int, default=32)
    ap.add_argument("--py", type=int, default=96)
    ap.add_argument("--px", type=int, default=96)
    ap.add_argument("--stride_z", type=int, default=16)
    ap.add_argument("--stride_y", type=int, default=48)
    ap.add_argument("--stride_x", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="patch3d")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.roi_json, "r") as f:
        rois = json.load(f)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print("[predict] device:", device)

    # load model
    model = UNet3DSmall(in_ch=3, base=16, out_ch=1).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    for idx in range(args.n):
        img_path = os.path.join(args.data_dir, f"image_test{idx:03d}.npy")
        img = np.load(img_path).astype(np.float32)  # (C,Z,Y,X)

        roi = load_roi(rois, "test", idx)
        if roi is None:
            # fallback whole volume
            C, Z, Y, X = img.shape
            roi = (0, Z, 0, Y, 0, X)

        prob = sliding_window_predict(
            img, model, roi,
            pz=args.pz, py=args.py, px=args.px,
            stride_z=args.stride_z, stride_y=args.stride_y, stride_x=args.stride_x,
            device=device
        )

        out_path = os.path.join(args.out_dir, f"pred_{args.tag}_id{idx:03d}.npy")
        np.save(out_path, prob.astype(np.float32))
        print("[saved]", out_path, "mean", float(prob.mean()))

    print("[done] wrote", args.n, "pred files to", args.out_dir)


if __name__ == "__main__":
    main()
