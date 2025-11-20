import os
import torch
import numpy as np

# -------------------------------
# Settings
# -------------------------------
FOLDER = "./data/prostate-data"
RESULT = "./result"
os.makedirs(RESULT, exist_ok=True)

use_cuda = torch.cuda.is_available()


# -------------------------------
# Smaller U-Net
# -------------------------------
class UNet(torch.nn.Module):
    def __init__(self, ch_in=1, ch_out=1, base=16):
        super().__init__()
        f = base

        self.enc1 = self.block(ch_in, f)
        self.enc2 = self.block(f, f*2)
        self.enc3 = self.block(f*2, f*4)

        self.pool = torch.nn.MaxPool3d(2)

        self.bottleneck = self.block(f*4, f*8)

        self.up3 = torch.nn.ConvTranspose3d(f*8, f*4, 2, 2)
        self.dec3 = self.block(f*8, f*4)

        self.up2 = torch.nn.ConvTranspose3d(f*4, f*2, 2, 2)
        self.dec2 = self.block(f*4, f*2)

        self.up1 = torch.nn.ConvTranspose3d(f*2, f, 2, 2)
        self.dec1 = self.block(f*2, f)

        self.out = torch.nn.Conv3d(f, 1, 1)

    def block(self, ni, nf):
        return torch.nn.Sequential(
            torch.nn.Conv3d(ni, nf, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(nf, nf, 3, padding=1),
            torch.nn.ReLU()
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


# -------------------------------
# Dice Loss
# -------------------------------
def dice_loss(pred, target, eps=1e-6):
    num = (pred * target).sum((2,3,4)) * 2
    den = pred.sum((2,3,4)) + target.sum((2,3,4)) + eps
    return 1 - (num / den).mean()


# -------------------------------
# Dataset loader
# -------------------------------
class NPYDataset(torch.utils.data.Dataset):
    def __init__(self, folder, split):
        self.folder = folder
        self.split = split
        self.files = sorted([f for f in os.listdir(folder) if f.startswith(f"image_{split}")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.folder, self.files[idx])).astype(np.float32)
        img = torch.tensor(img).unsqueeze(0)

        if self.split == "train":
            lab = np.load(os.path.join(self.folder, self.files[idx].replace("image", "label")))
            lab = torch.tensor(lab).unsqueeze(0).float()
            return img, lab
        else:
            return img, idx


# -------------------------------
# Training
# -------------------------------
if __name__ == "__main__":

    train_ds = NPYDataset(FOLDER, "train")
    test_ds  = NPYDataset(FOLDER, "test")

    if len(test_ds) == 0:
        raise RuntimeError("❌ No test files found — run prepare_prostate_data.py first.")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=1, shuffle=True, num_workers=0)

    model = UNet()
    if use_cuda:
        model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_steps = 100   # much faster
    print_every = 20
    test_every = 50

    step = 0

    while step < total_steps:
        for imgs, labs in train_loader:
            step += 1

            if use_cuda:
                imgs, labs = imgs.cuda(), labs.cuda()

            opt.zero_grad()
            preds = model(imgs)
            loss = dice_loss(preds, labs)
            loss.backward()
            opt.step()

            if step % print_every == 0:
                print(f"[Step {step}] Loss = {loss.item():.4f}")

            if step % test_every == 0 or step == total_steps:
                imgs_t, ids = next(iter(test_loader))
                if use_cuda:
                    imgs_t = imgs_t.cuda()

                out = model(imgs_t).detach().cpu().numpy()

                save_path = f"{RESULT}/pred_{ids[0]:03d}.npy"
                np.save(save_path, out[0,0])
                print(f"Saved test prediction → {save_path}")

            if step >= total_steps:
                break

    torch.save(model.state_dict(), f"{RESULT}/model_small.pth")
    print("✔ Training complete.")
