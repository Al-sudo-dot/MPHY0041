import os
import json
import numpy as np
import torch

PROGRESS_TEST_ID = 0  # change this to any test case you want
LESION_DATA = "./data/lesion-data"
ROI_JSON    = "./result/prostate_rois.json"
RESULT      = "./result_lesion_roi"
os.makedirs(RESULT, exist_ok=True)

use_cuda = torch.cuda.is_available()

# -----------------------
# shape helpers (robust to odd ROI sizes)
# -----------------------
def _center_crop_like(src, ref):
    """
    Center-crop src so its spatial dims (D,H,W) match ref.
    Works on 5D tensors: (B,C,D,H,W)
    """
    assert src.ndim == 5 and ref.ndim == 5, (src.shape, ref.shape)
    _, _, ds, hs, ws = src.shape
    _, _, dr, hr, wr = ref.shape

    d1 = (ds - dr) // 2
    h1 = (hs - hr) // 2
    w1 = (ws - wr) // 2
    return src[:, :, d1:d1 + dr, h1:h1 + hr, w1:w1 + wr]

def _center_crop_to_common(a, b):
    """
    Center-crop a and b so they share the same (D,H,W).
    Returns (a_c, b_c). 5D tensors only.
    """
    assert a.ndim == 5 and b.ndim == 5, (a.shape, b.shape)

    da, ha, wa = a.shape[2], a.shape[3], a.shape[4]
    db, hb, wb = b.shape[2], b.shape[3], b.shape[4]

    d = min(da, db)
    h = min(ha, hb)
    w = min(wa, wb)

    def _crop(x, d, h, w):
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        d1 = (D - d) // 2
        h1 = (H - h) // 2
        w1 = (W - w) // 2
        return x[:, :, d1:d1 + d, h1:h1 + h, w1:w1 + w]

    return _crop(a, d, h, w), _crop(b, d, h, w)

# -----------------------
# 3D UNet (ch_in=3 for lesion)
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

        return self.out(d1)
# -----------------------
# Dice loss (shape-safe)
# -----------------------
def dice_loss(logits, target, eps=1e-6, return_parts=False):
    """
    Combined BCEWithLogits + Dice loss.
    Expects raw logits (NO sigmoid in model forward).
    """
    # align shapes first (BCE requires exact match)
    if logits.shape[2:] != target.shape[2:]:
        logits, target = _center_crop_to_common(logits, target)

    target = target.float()

    # BCE
    pos_weight = torch.tensor([10.0], device=logits.device)
    bce_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce = bce_fn(logits, target)

    # Dice
    probs = torch.sigmoid(logits)
    num = (probs * target).sum((2, 3, 4)) * 2.0
    den = probs.sum((2, 3, 4)) + target.sum((2, 3, 4)) + eps
    dice = 1.0 - (num / den).mean()

    total = 0.3 * bce + 0.7 * dice
    if return_parts:
        return total, bce.detach(), dice.detach()
    return total

# -----------------------
# ROI helpers
# -----------------------
def crop_czyx(img_czyx, bb):
    z0, z1, y0, y1, x0, x1 = bb
    return img_czyx[:, z0:z1, y0:y1, x0:x1]

def crop_zyx(vol_zyx, bb):
    z0, z1, y0, y1, x0, x1 = bb
    return vol_zyx[z0:z1, y0:y1, x0:x1]

def paste_into_full(full_shape, bb, roi_pred):
    z0, z1, y0, y1, x0, x1 = bb
    out = np.zeros(full_shape, dtype=np.float32)

    # Safety in case roi_pred got cropped slightly smaller than bb
    dz = min(z1 - z0, roi_pred.shape[0])
    dy = min(y1 - y0, roi_pred.shape[1])
    dx = min(x1 - x0, roi_pred.shape[2])

    out[z0:z0+dz, y0:y0+dy, x0:x0+dx] = roi_pred[:dz, :dy, :dx]
    return out

# -----------------------
# Dataset
# -----------------------
class LesionROIDataset(torch.utils.data.Dataset):
    def __init__(self, folder, split, rois):
        self.folder = folder
        self.split = split
        self.rois = rois
        self.img_files = sorted(
            f for f in os.listdir(folder)
            if f.startswith(f"image_{split}") and f.endswith(".npy")
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        lab_name = img_name.replace("image_", "label_")

        img_full = np.load(os.path.join(self.folder, img_name)).astype(np.float32)  # (3,Z,Y,X)
        lab_full = np.load(os.path.join(self.folder, lab_name)).astype(np.float32)  # (Z,Y,X)

        bb = self.rois[img_name]

        img_roi = crop_czyx(img_full, bb)               # (3,z,y,x)
        lab_roi = crop_zyx(lab_full, bb)                # (z,y,x)

        img_t = torch.from_numpy(img_roi)               # (3,z,y,x)
        lab_t = torch.from_numpy(lab_roi).unsqueeze(0)  # (1,z,y,x)

        return img_t, lab_t, img_name, lab_full.shape

@torch.no_grad()
def predict_full(model, img_full_czyx, bb):
    roi = crop_czyx(img_full_czyx, bb)  # (3,z,y,x)
    x = torch.from_numpy(roi.astype(np.float32)).unsqueeze(0)  # (1,3,z,y,x)
    if use_cuda:
        x = x.cuda()

    logits = model(x)
    prob_roi = torch.sigmoid(logits)[0,0].detach().cpu().numpy()

    return paste_into_full(img_full_czyx.shape[1:], bb, prob_roi)

def main():
    if not os.path.exists(ROI_JSON):
        raise FileNotFoundError(f"Missing ROI json: {ROI_JSON}")

    rois = json.load(open(ROI_JSON, "r"))

    train_ds = LesionROIDataset(LESION_DATA, "train", rois)
    test_ds  = LesionROIDataset(LESION_DATA, "test",  rois)

    if len(train_ds) == 0:
        raise RuntimeError("No lesion train files found (image_train*.npy).")
    if len(test_ds) == 0:
        raise RuntimeError("No lesion test files found (image_test*.npy).")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    model = UNet(ch_in=3)
    if use_cuda:
        model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_steps = 1200
    print_every = 50
    test_every  = 100

    step = 0
    while step < total_steps:
        for img_t, lab_t, _, _ in train_loader:
            step += 1

            # DataLoader gives (B,3,z,y,x) and (B,1,z,y,x)
            if use_cuda:
                img_t = img_t.cuda()
                lab_t = lab_t.cuda()

            opt.zero_grad()
            pred = model(img_t)

            # Optional debug (comment out if noisy)
            # if pred.shape[2:] != lab_t.shape[2:]:
            #     print("SHAPE MISMATCH:", pred.shape, lab_t.shape)

            loss, bce_v, dice_v = dice_loss(pred, lab_t, return_parts=True)
            loss.backward()
            opt.step()

            if step % print_every == 0:
                print(f"[Step {step}] loss={loss.item():.4f}  bce={float(bce_v):.4f}  dice={float(dice_v):.4f}")

            if step % test_every == 0 or step == total_steps:
                img_name2 = f"image_test{PROGRESS_TEST_ID:03d}.npy"
                img_full = np.load(os.path.join(LESION_DATA, img_name2)).astype(np.float32)
                bb = rois[img_name2]
                model.eval()
                full_prob = predict_full(model, img_full, bb)
                model.train()

                save_path = os.path.join(RESULT, f"progress_pred_step{step:04d}_id{PROGRESS_TEST_ID:03d}.npy")
                np.save(save_path, full_prob)
                print(f"Saved test prediction → {save_path}")

            if step >= total_steps:
                break

    torch.save(model.state_dict(), os.path.join(RESULT, "model_lesion_roi.pth"))
    print("✔ Lesion ROI training complete.")

if __name__ == "__main__":
    main()
