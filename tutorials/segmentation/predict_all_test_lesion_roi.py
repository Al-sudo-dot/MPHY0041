import os, glob, argparse, json
import numpy as np
import torch

# Reuse your exact model + ROI helpers from your training script
import train_lesion_roi_posw10_backup as T


@torch.no_grad()
def predict_full_prob(model, img_full_czyx, bb, use_cuda=True):
    """
    Crop ROI from full (C,Z,Y,X), run model on ROI, paste back into full (Z,Y,X).
    Returns probability map in full volume coordinates.
    """
    roi = T.crop_czyx(img_full_czyx, bb)  # (C,z,y,x)
    x = torch.from_numpy(roi.astype(np.float32)).unsqueeze(0)  # (1,C,z,y,x)
    if use_cuda:
        x = x.cuda()

    logits = model(x)  # (1,1,z,y,x)
    prob_roi = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    full_shape_zyx = img_full_czyx.shape[1:]  # (Z,Y,X)
    return T.paste_into_full(full_shape_zyx, bb, prob_roi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model_lesion_roi.pth (state_dict)")
    ap.add_argument("--data_dir", default="./data/lesion-data")
    ap.add_argument("--roi_json", default="./result/prostate_rois.json")
    ap.add_argument("--out_dir", default="./result_lesion_roi")
    ap.add_argument("--step", type=int, default=1200, help="used in filename pred_step{step:04d}_idXYZ.npy")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    use_cuda = (args.device == "cuda" and torch.cuda.is_available())
    print("[predict] device:", "cuda" if use_cuda else "cpu")

    if not os.path.exists(args.roi_json):
        raise FileNotFoundError(f"Missing ROI json: {args.roi_json}")
    rois = json.load(open(args.roi_json, "r"))

    # Build model exactly like training
    model = T.UNet(ch_in=3)

    state = torch.load(args.model, map_location="cuda" if use_cuda else "cpu")
    # If state dict was saved directly, this works.
    # If you ever saved a checkpoint dict, tell me and we’ll adjust one line.
    model.load_state_dict(state)

    if use_cuda:
        model.cuda()
    model.eval()

    img_paths = sorted(glob.glob(os.path.join(args.data_dir, "image_test*.npy")))
    if not img_paths:
        raise RuntimeError(f"No test images found in {args.data_dir} matching image_test*.npy")

    for p in img_paths:
        base = os.path.basename(p)  # image_testXYZ.npy
        idx_str = base.replace("image_test", "").replace(".npy", "")
        pid = int(idx_str)

        img_full = np.load(p).astype(np.float32)  # (C,Z,Y,X)
        if base not in rois:
            raise KeyError(f"ROI json missing key for {base}")
        bb = rois[base]

        full_prob = predict_full_prob(model, img_full, bb, use_cuda=use_cuda)

        out_path = os.path.join(args.out_dir, f"pred_step{args.step:04d}_id{pid:03d}.npy")
        np.save(out_path, full_prob)

        if pid % 10 == 0:
            print("[saved]", out_path, "mean", float(full_prob.mean()))

    print("[done] wrote", len(img_paths), "pred files to", args.out_dir)


if __name__ == "__main__":
    main()
