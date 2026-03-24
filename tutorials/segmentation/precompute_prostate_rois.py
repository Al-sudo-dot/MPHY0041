import os
import json
import numpy as np
import torch

# -----------------------
# Paths
# -----------------------
LESION_DATA = "./data/lesion-data"
CKPT        = "./result/model_small.pth"
OUT_JSON    = "./result/prostate_rois.json"

THR = 0.5
PAD = (4, 24, 24)

use_cuda = torch.cuda.is_available()


# -----------------------
# 3D UNet (same as prostate)
# -----------------------
class UNet(torch.nn.Module):
    def __init__(self, ch_in=1, ch_out=1, base=16):
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

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))


# -----------------------
# Helpers
# -----------------------
def bbox_from_mask(mask, pad):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    pz, py, px = pad

    z0 = max(0, z0 - pz)
    y0 = max(0, y0 - py)
    x0 = max(0, x0 - px)

    z1 = min(mask.shape[0], z1 + pz)
    y1 = min(mask.shape[1], y1 + py)
    x1 = min(mask.shape[2], x1 + px)

    return [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)]


@torch.no_grad()
def predict_prostate(model, vol):
    x = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if use_cuda:
        x = x.cuda()
    prob = model(x)[0, 0].detach().cpu().numpy()
    return prob


def load_first_channel(path):
    arr = np.load(path).astype(np.float32)

    if arr.ndim == 4:  # (C,Z,Y,X)
        return arr[0]

    if arr.ndim == 3:
        return arr

    raise ValueError(f"Unexpected shape {arr.shape}")


# -----------------------
# Main
# -----------------------
def main():

    if not os.path.isdir(LESION_DATA):
        raise FileNotFoundError("Lesion data folder not found.")

    if not os.path.exists(CKPT):
        raise FileNotFoundError("Prostate model checkpoint not found.")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    model = UNet()
    if use_cuda:
        model.cuda()

    state = torch.load(
        CKPT,
        map_location="cuda" if use_cuda else "cpu",
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    img_files = sorted(
        f for f in os.listdir(LESION_DATA)
        if f.startswith("image_") and f.endswith(".npy")
    )

    print("Found", len(img_files), "lesion volumes")

    rois = {}
    empty_count = 0

    for i, f in enumerate(img_files):

        print(f"[{i+1}/{len(img_files)}] {f}", flush=True)

        path = os.path.join(LESION_DATA, f)

        vol = load_first_channel(path)
        prob = predict_prostate(model, vol)
        mask = (prob >= THR).astype(np.uint8)

        np.save("prostate_mask_example.npy", mask)
        print("Saved prostate mask: prostate_mask_example.npy")

        bb = bbox_from_mask(mask, PAD)

        if bb is None:
            empty_count += 1
            bb = [0, vol.shape[0], 0, vol.shape[1], 0, vol.shape[2]]

        rois[f] = bb

    with open(OUT_JSON, "w") as fp:
        json.dump(rois, fp, indent=2)

    print("\nSaved ROIs →", OUT_JSON)
    print("Empty masks:", empty_count)


if __name__ == "__main__":
    main()
