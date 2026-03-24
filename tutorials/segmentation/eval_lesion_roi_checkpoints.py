import os
import glob
import json
import argparse
import numpy as np
import torch

LESION_DATA = "./data/lesion-data"
ROI_JSON    = "./result/prostate_rois.json"
RESULT_DIR  = "./result_lesion_roi"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# -----------------------
# Safe center-crop to common spatial size
# -----------------------
def _center_crop_to_common(a, b):
    """
    Center-crop tensors a and b so their spatial shapes match.
    a, b: torch tensors shaped (B,C,Z,Y,X)
    """
    az, ay, ax = a.shape[-3:]
    bz, by, bx = b.shape[-3:]
    tz, ty, tx = min(az, bz), min(ay, by), min(ax, bx)

    def crop(x, tz, ty, tx):
        z, y, xw = x.shape[-3:]
        z0 = (z - tz) // 2
        y0 = (y - ty) // 2
        x0 = (xw - tx) // 2
        return x[..., z0:z0+tz, y0:y0+ty, x0:x0+tx]

    return crop(a, tz, ty, tx), crop(b, tz, ty, tx)


# -----------------------
# ROI helpers
# -----------------------
def crop_czyx(img_czyx, bb):
    z0, z1, y0, y1, x0, x1 = bb
    return img_czyx[:, z0:z1, y0:y1, x0:x1]

def crop_zyx(vol_zyx, bb):
    z0, z1, y0, y1, x0, x1 = bb
    return vol_zyx[z0:z1, y0:y1, x0:x1]

def paste_into_full(full_shape_zyx, bb, roi_zyx):
    z0, z1, y0, y1, x0, x1 = bb
    out = np.zeros(full_shape_zyx, dtype=np.float32)
    out[z0:z1, y0:y1, x0:x1] = roi_zyx
    return out


def shrink_bb_to_match(in_shape_zyx, out_shape_zyx, bb):
    """
    If model output ROI is smaller than input ROI (due to internal center-cropping),
    shrink the bbox by equal-ish margins so paste region matches out_shape_zyx.
    """
    iz, iy, ix = in_shape_zyx
    oz, oy, ox = out_shape_zyx

    dz = max(0, iz - oz)
    dy = max(0, iy - oy)
    dx = max(0, ix - ox)

    dz0 = dz // 2
    dy0 = dy // 2
    dx0 = dx // 2

    dz1 = dz - dz0
    dy1 = dy - dy0
    dx1 = dx - dx0

    z0, z1, y0, y1, x0, x1 = bb
    return [z0 + dz0, z1 - dz1,
            y0 + dy0, y1 - dy1,
            x0 + dx0, x1 - dx1]


# -----------------------
# Postprocess: keep top-K components (for multiple lesions)
# -----------------------
def keep_topk_components(mask_zyx, k=3, min_vox=20):
    """
    mask_zyx: (Z,Y,X) uint8/bool
    Keeps top-k largest connected components (26-connectivity).
    If scipy is missing, returns mask unchanged.
    """
    try:
        import scipy.ndimage as ndi
    except Exception:
        return mask_zyx.astype(np.uint8)

    mask = mask_zyx.astype(np.bool_)
    if mask.sum() == 0:
        return mask_zyx.astype(np.uint8)

    structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
    lab, n = ndi.label(mask, structure=structure)
    if n == 0:
        return mask_zyx.astype(np.uint8)

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0

    if min_vox is not None and min_vox > 0:
        sizes[sizes < int(min_vox)] = 0

    if sizes.max() == 0:
        return np.zeros_like(mask_zyx, dtype=np.uint8)

    keep_ids = np.argsort(sizes)[-int(k):]
    out = np.isin(lab, keep_ids)
    return out.astype(np.uint8)


# -----------------------
# Dice (binary)
# -----------------------
def dice_binary(pred_zyx, gt_zyx, eps=1e-6):
    pred = pred_zyx.astype(np.bool_)
    gt   = gt_zyx.astype(np.bool_)
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter + eps) / (pred.sum() + gt.sum() + eps)


# -----------------------
# 3D UNet (must match training) — outputs LOGITS
# -----------------------
class UNet(torch.nn.Module):
    def __init__(self, ch_in=3, ch_out=1, base=16):
        super().__init__()
        f = base

        self.enc1 = self.block(ch_in, f)
        self.enc2 = self.block(f, f * 2)
        self.enc3 = self.block(f * 2, f * 4)

        self.pool = torch.nn.MaxPool3d(2)

        self.bottleneck = self.block(f * 4, f * 8)

        self.up3 = torch.nn.ConvTranspose3d(f * 8, f * 4, 2, 2)
        self.dec3 = self.block(f * 8, f * 4)

        self.up2 = torch.nn.ConvTranspose3d(f * 4, f * 2, 2, 2)
        self.dec2 = self.block(f * 4, f * 2)

        self.up1 = torch.nn.ConvTranspose3d(f * 2, f, 2, 2)
        self.dec1 = self.block(f * 2, f)

        self.out = torch.nn.Conv3d(f, ch_out, 1)

    def block(self, ni, nf):
        return torch.nn.Sequential(
            torch.nn.Conv3d(ni, nf, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(nf, nf, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        u3 = self.up3(b)
        u3, e3c = _center_crop_to_common(u3, e3)
        d3 = self.dec3(torch.cat([u3, e3c], dim=1))

        u2 = self.up2(d3)
        u2, e2c = _center_crop_to_common(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2c], dim=1))

        u1 = self.up1(d2)
        u1, e1c = _center_crop_to_common(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1c], dim=1))

        return self.out(d1)  # logits


@torch.no_grad()
def predict_full_prob_and_bb(model, img_full_czyx, bb):
    """
    Returns:
      prob_full: (Z,Y,X) float32 in [0,1] with ROI pasted into full volume
      bb_adj: adjusted bbox matching model output spatial size
    """
    roi_in = crop_czyx(img_full_czyx, bb)     # (3, z, y, x)
    in_shape = roi_in.shape[1:]               # (z, y, x)

    x = torch.from_numpy(roi_in.astype(np.float32)).unsqueeze(0).to(device)  # (1,3,z,y,x)
    logits = model(x)
    prob_roi = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)  # (z2,y2,x2)

    bb_adj = bb
    if tuple(prob_roi.shape) != tuple(in_shape):
        bb_adj = shrink_bb_to_match(in_shape, prob_roi.shape, bb)

    prob_full = paste_into_full(img_full_czyx.shape[1:], bb_adj, prob_roi)
    return prob_full, bb_adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default=RESULT_DIR)
    ap.add_argument("--roi_json", default=ROI_JSON)
    ap.add_argument("--data_dir", default=LESION_DATA)
    ap.add_argument("--k", type=int, default=3, help="keep top-k connected components")
    ap.add_argument("--min_vox", type=int, default=20, help="drop components smaller than this")
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    if not os.path.exists(args.roi_json):
        raise FileNotFoundError(f"Missing ROI json: {args.roi_json}")
    rois = json.load(open(args.roi_json, "r"))

    ckpts = sorted(glob.glob(os.path.join(args.result_dir, "model_lesion_roi.pth")))
    if len(ckpts) == 0:
        # fallback: any .pth in the folder
        ckpts = sorted(glob.glob(os.path.join(args.result_dir, "*.pth")))
    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoints found in {args.result_dir}")

    test_imgs = sorted(glob.glob(os.path.join(args.data_dir, "image_test*.npy")))
    if len(test_imgs) == 0:
        raise RuntimeError("No test images found (image_test*.npy).")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    print(f"Found checkpoints: {ckpts}")

    for ck in ckpts:
        model = UNet(ch_in=3).to(device)

        # avoid future warning + safer load if supported
        try:
            sd = torch.load(ck, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(ck, map_location=device)

        model.load_state_dict(sd)
        model.eval()

        scores = {thr: [] for thr in thresholds}

        for img_path in test_imgs:
            img_name = os.path.basename(img_path)
            lab_name = img_name.replace("image_", "label_")
            lab_path = os.path.join(args.data_dir, lab_name)

            img_full = np.load(img_path).astype(np.float32)  # (3,Z,Y,X)
            gt_full  = np.load(lab_path).astype(np.uint8)    # (Z,Y,X)

            if img_name not in rois:
                raise KeyError(f"ROI not found for {img_name} in {args.roi_json}")
            bb = rois[img_name]

            prob_full, bb_adj = predict_full_prob_and_bb(model, img_full, bb)

            # FAIR EVAL: compute Dice on the same adjusted ROI region
            prob_roi = crop_zyx(prob_full, bb_adj)  # matches model output region
            gt_roi   = crop_zyx(gt_full,  bb_adj)

            for thr in thresholds:
                pred_roi = (prob_roi >= thr).astype(np.uint8)
                pred_roi = keep_topk_components(pred_roi, k=args.k, min_vox=args.min_vox)
                d = dice_binary(pred_roi, gt_roi)
                scores[thr].append(d)

        msg = []
        for thr in thresholds:
            msg.append(f"thr {thr:.1f}: {np.mean(scores[thr]):.3f}")
        print(f"{os.path.basename(ck)} => " + " | ".join(msg))


if __name__ == "__main__":
    main()
