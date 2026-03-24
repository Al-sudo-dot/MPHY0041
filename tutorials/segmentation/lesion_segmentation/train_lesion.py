# tutorials/segmentation/lesion_segmentation/train_lesion.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# PATHS (run from MPHY0041 root)
# ============================================================
DATA_DIR = "/cs/student/projects4/misc/alukic/outputs/lesion-data"
RESULT_DIR = os.environ.get("RESULT_DIR", "/cs/student/projects4/misc/alukic/results/lesion")
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# STEP-BASED TRAINING SETTINGS (matches prostate style)
# ============================================================
BATCH_SIZE   = 2
TOTAL_STEPS  = int(os.environ.get("TOTAL_STEPS", "100"))          # like prostate
PRINT_EVERY  = int(os.environ.get("PRINT_EVERY", "20"))          # prints [Step XX]
SAVE_EVERY   = int(os.environ.get("SAVE_EVERY", "50"))           # saves pred_stepXXXX_id000.npy
LR           = 1e-3

# lesion imbalance
POS_WEIGHT = float(os.environ.get("POS_WEIGHT", "10"))

# patching
PATCH_SIZE = (32,96,96)   # (Z,Y,X)
POS_PATCH_PROB = float(os.environ.get("POS_PATCH_PROB", "0.8"))

# keep simple / stable on mac
NUM_WORKERS = 0

# ============================================================
# PATCH SAMPLING
# ============================================================
def sample_patch(X, y, patch_size, force_positive=False, prostate_mask=None):
    """
    X: (C,Z,Y,X)
    y: (Z,Y,X) binary {0,1}
    prostate_mask: (Z,Y,X) binary {0,1} used to sample negatives within prostate
    """
    C, Dz, Dy, Dx = X.shape
    pz, py, px = patch_size

    def clamp(c, P, D):
        half = P // 2
        return max(half, min(int(c), D - half - 1))

    # Choose a center voxel
    if force_positive and y.sum() > 0:
        pos = np.array(np.where(y == 1))
        cz, cy, cx = pos[:, np.random.randint(pos.shape[1])]
    else:
        # Prefer sampling from prostate region if provided
        if prostate_mask is not None and prostate_mask.sum() > 0:
            pts = np.array(np.where(prostate_mask > 0))
            cz, cy, cx = pts[:, np.random.randint(pts.shape[1])]
        else:
            cz = np.random.randint(0, Dz)
            cy = np.random.randint(0, Dy)
            cx = np.random.randint(0, Dx)

    cz, cy, cx = clamp(cz, pz, Dz), clamp(cy, py, Dy), clamp(cx, px, Dx)

    z0, z1 = cz - pz//2, cz + pz//2
    y0, y1 = cy - py//2, cy + py//2
    x0, x1 = cx - px//2, cx + px//2

    Xp = X[:, z0:z1, y0:y1, x0:x1]
    yp = y[z0:z1, y0:y1, x0:x1]
    return Xp, yp

# ============================================================
# DATASET: loads volumes, returns random patches
# ============================================================
class LesionPatchDataset(Dataset):
    def __init__(self, split: str):
        self.split = split
        self.image_files = sorted([
            f for f in os.listdir(DATA_DIR)
            if f.startswith(f"image_{split}") and f.endswith(".npy")
        ])
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No {split} files found in {DATA_DIR}")

    def __len__(self):
        # DataLoader can cycle forever; length isn't critical for step-based training
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx % len(self.image_files)]
        lab_file = img_file.replace("image_", "label_")

        X = np.load(os.path.join(DATA_DIR, img_file)).astype(np.float32)  # (3,Z,Y,X)
        y = np.load(os.path.join(DATA_DIR, lab_file)).astype(np.float32)  # (Z,Y,X) or (1,Z,Y,X)

        if y.ndim == 4:
            y = y[0]
        y = (y > 0).astype(np.uint8)

        # use channel 2 as prostate mask (3rd channel)
        prostate = (X[2] > 0).astype(np.uint8) if X.shape[0] >= 3 else None

        force_pos = (np.random.rand() < POS_PATCH_PROB)

        # try harder to get lesion-containing patch when forcing positive
        for _ in range(10):
            Xp, yp = sample_patch(
                X,
                y,
                PATCH_SIZE,
                force_positive=force_pos,
                prostate_mask=prostate,
            )
            if (not force_pos) or (yp.sum() > 0):
                break

        Xp = torch.from_numpy(Xp).float()              # (3,pz,py,px)
        yp = torch.from_numpy(yp[None, ...]).float()   # (1,pz,py,px)
        return Xp, yp


# ============================================================
# MODEL: small 3D U-Net (CPU)
# ============================================================
class UNet(torch.nn.Module):
    def __init__(self, ch_in=3, base=16):
        super().__init__()
        f = base
        self.enc1 = self.block(ch_in, f)
        self.enc2 = self.block(f, f*2)
        self.enc3 = self.block(f*2, f*4)
        self.pool = torch.nn.MaxPool3d(2)
        self.bott = self.block(f*4, f*8)
        self.up3  = torch.nn.ConvTranspose3d(f*8, f*4, 2, 2)
        self.dec3 = self.block(f*8, f*4)
        self.up2  = torch.nn.ConvTranspose3d(f*4, f*2, 2, 2)
        self.dec2 = self.block(f*4, f*2)
        self.up1  = torch.nn.ConvTranspose3d(f*2, f, 2, 2)
        self.dec1 = self.block(f*2, f)
        self.out  = torch.nn.Conv3d(f, 1, 1)

    def block(self, ni, nf):
        return torch.nn.Sequential(
            torch.nn.Conv3d(ni, nf, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(nf, nf, 3, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)  # logits

# ============================================================
# COMBINED LOSS: BCEWithLogits + Dice
# ============================================================
def dice_loss_with_logits(logits, target, mask=None, eps=1e-6):
    """
    logits: (B,1,D,H,W)
    target: (B,1,D,H,W)
    mask:   (B,1,D,H,W) with 1 inside prostate, 0 outside
    """
    p = torch.sigmoid(logits)

    if mask is not None:
        p = p * mask
        target = target * mask

    num = (p * target).sum((2, 3, 4)) * 2
    den = p.sum((2, 3, 4)) + target.sum((2, 3, 4)) + eps
    return 1 - (num / den).mean()


def combined_loss(logits, target, mask=None):
    """
    Masked BCEWithLogits + Masked Dice
    """
    pos_w = torch.tensor(POS_WEIGHT, device=logits.device)

    # BCE per-voxel (no reduction yet)
    bce_map = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_w, reduction="none"
    )

    if mask is not None:
        bce = (bce_map * mask).sum() / (mask.sum() + 1e-6)
    else:
        bce = bce_map.mean()

    dsc = dice_loss_with_logits(logits, target, mask=mask)

    return 0.5 * bce + 0.5 * dsc

# ============================================================
# FULL-VOLUME PREDICTION (for saving like prostate)
# ============================================================
@torch.no_grad()
def predict_full_volume(model, volume3ch):
    """
    volume3ch: numpy (3, Z, Y, X)
    returns: numpy (Z, Y, X) prediction (0/1) using sigmoid>0.5
    """
    model.eval()
    x = torch.from_numpy(volume3ch[None, ...]).float()  # (1,3,Z,Y,X)
    logits = model(x)                                   # (1,1,Z,Y,X)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred = (prob >= 0.6).astype(np.uint8)
    return pred

# ============================================================
# MAIN: step-based training loop
# ============================================================
def main():
    # training data (patches)
    train_ds = LesionPatchDataset("train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # pick ONE fixed test volume to save predictions for (id000 like prostate)
    test_images = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("image_test") and f.endswith(".npy")])
    if len(test_images) == 0:
        raise FileNotFoundError(f"No image_testXXX.npy found in {DATA_DIR}")

    test_img_file = test_images[0]  # "id000" equivalent
    test_id = 0

    model = UNet()  # CPU by default
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # step-based training
    step = 0
    loader_iter = iter(train_loader)

    while step < TOTAL_STEPS:
        try:
            X, Y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            X, Y = next(loader_iter)

        model.train()
        opt.zero_grad(set_to_none=True)

        logits = model(X)

        # prostate mask comes from channel 2 (3rd channel)
        mask = (X[:, 2:3] > 0).float()

        loss = combined_loss(logits, Y, mask)

        loss.backward()
        opt.step()

        step += 1

        if step % PRINT_EVERY == 0:
            print(f"[Step {step}] Loss = {loss.item():.4f}")

        if step % SAVE_EVERY == 0:
            test_vol = np.load(os.path.join(DATA_DIR, test_img_file)).astype(np.float32)  # (3,Z,Y,X)

            pred = predict_full_volume(model, test_vol)

            save_path = os.path.join(RESULT_DIR, f"pred_step{step:04d}_id{test_id:03d}.npy")
            np.save(save_path, pred)
            print(f"Saved test prediction -> {save_path}")

    # save final model
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, "lesion_model_last.pth"))
    print("✓ Training complete.")

if __name__ == "__main__":
    main()
