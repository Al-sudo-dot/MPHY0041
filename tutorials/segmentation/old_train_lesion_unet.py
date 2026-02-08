import os
import torch
import numpy as np
import torch.nn.functional as F

# -------------------------------
# Settings
# -------------------------------
FOLDER = "./data/lesion-roi-data"   # <-- CHANGED (ROI dataset)
RESULT = "./result_lesion_roi"
os.makedirs(RESULT, exist_ok=True)

use_cuda = torch.cuda.is_available()

# Fixed input size for the network.
# ROI crops will vary per patient, so we standardize them.
# This size is usually safe given your original volume was (160,160,64).
FIXED_SIZE = (128, 128, 64)  # (H, W, D)


# -------------------------------
# Pad/crop to fixed size
# -------------------------------
def pad_or_center_crop_3d(x, target_shape=FIXED_SIZE):
    """
    Make x exactly target_shape by:
      - center-cropping if x is larger
      - zero-padding if x is smaller
    Works for both image and mask.
    """
    th, tw, td = target_shape
    H, W, D = x.shape

    # --- center crop if larger ---
    hs = max(0, (H - th) // 2)
    ws = max(0, (W - tw) // 2)
    ds = max(0, (D - td) // 2)

    x_c = x[hs:hs + min(th, H), ws:ws + min(tw, W), ds:ds + min(td, D)]

    # --- pad if smaller ---
    out = np.zeros(target_shape, dtype=x.dtype)
    h, w, d = x_c.shape
    oh = (th - h) // 2
    ow = (tw - w) // 2
    od = (td - d) // 2
    out[oh:oh + h, ow:ow + w, od:od + d] = x_c
    return out


# -------------------------------
# Smaller 3D U-Net (outputs LOGITS)
# -------------------------------
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

        return self.out(d1)  # logits


# -------------------------------
# Loss: Dice (from logits) + BCEWithLogits
# -------------------------------
def dice_loss_from_logits(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    num = (pred * target).sum((2, 3, 4)) * 2.0
    den = pred.sum((2, 3, 4)) + target.sum((2, 3, 4)) + eps
    return 1.0 - (num / den).mean()

def combo_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dsc = dice_loss_from_logits(logits, target)
    return dsc + 0.5 * bce


# -------------------------------
# Dataset loader (ROI volumes -> fixed size)
# -------------------------------
class NPYDataset(torch.utils.data.Dataset):
    def __init__(self, folder, split, target_shape=FIXED_SIZE):
        self.folder = folder
        self.split = split
        self.target_shape = target_shape
        self.files = sorted([f for f in os.listdir(folder) if f.startswith(f"image_{split}")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.folder, self.files[idx])).astype(np.float32)
        img = pad_or_center_crop_3d(img, self.target_shape)
        img = torch.from_numpy(img).unsqueeze(0)

        if self.split == "train":
            lab_name = self.files[idx].replace("image", "label")
            lab = np.load(os.path.join(self.folder, lab_name)).astype(np.float32)
            lab = pad_or_center_crop_3d(lab, self.target_shape)
            lab = torch.from_numpy(lab).unsqueeze(0)
            return img, lab
        else:
            return img, idx


# -------------------------------
# Training
# -------------------------------
if __name__ == "__main__":

    train_ds = NPYDataset(FOLDER, "train", target_shape=FIXED_SIZE)
    test_ds  = NPYDataset(FOLDER, "test",  target_shape=FIXED_SIZE)

    if len(train_ds) == 0:
        raise RuntimeError("❌ No train files found — run ROI prep first.")
    if len(test_ds) == 0:
        raise RuntimeError("❌ No test files found — run ROI prep first.")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=1, shuffle=False)

    model = UNet()
    if use_cuda:
        model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_steps = 5000
    print_every = 50
    test_every  = 100
    debug_every = 50

    step = 0

    while step < total_steps:
        for imgs, labs in train_loader:
            step += 1

            if use_cuda:
                imgs = imgs.cuda()
                labs = labs.cuda()

            opt.zero_grad()
            logits = model(imgs)

            if step % debug_every == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    print(f"[Step {step}] prob mean={probs.mean().item():.4f} max={probs.max().item():.4f}")

            loss = combo_loss(logits, labs)
            loss.backward()
            opt.step()

            if step % print_every == 0:
                print(f"[Step {step}] Loss = {loss.item():.4f}")

            if step % test_every == 0 or step == total_steps:
                model.eval()
                with torch.no_grad():
                    imgs_t, ids = next(iter(test_loader))
                    if use_cuda:
                        imgs_t = imgs_t.cuda()
                    out = torch.sigmoid(model(imgs_t)).cpu().numpy()
                model.train()

                save_path = os.path.join(RESULT, f"pred_step{step:04d}_id{int(ids[0]):03d}.npy")
                np.save(save_path, out[0, 0])
                print(f"Saved test prediction → {save_path}")

            if step >= total_steps:
                break

    torch.save(model.state_dict(), os.path.join(RESULT, "model_small.pth"))
    print("✔ Training complete.")
