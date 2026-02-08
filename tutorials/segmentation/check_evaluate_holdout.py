import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

# -------------------------
# Paths
# -------------------------
FOLDER = "./data/prostate-data"
RESULT = "./result"
MODEL_PATH = os.path.join(RESULT, "model_small.pth")

# -------------------------
# Load U-Net model
# -------------------------
from train_unet import UNet   # you already wrote this!

use_cuda = torch.cuda.is_available()

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
if use_cuda:
    model.cuda()

# -------------------------
# Utility Functions
# -------------------------
def dice_score(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).astype(np.float32)
    num = (pred_bin * target).sum() * 2.0
    den = pred_bin.sum() + target.sum() + eps
    return num / den

def hausdorff_distance(pred, target):
    """Simple Hausdorff distance using scipy (not symmetric)."""
    pred_pts = np.argwhere(pred > 0.5)
    gt_pts   = np.argwhere(target > 0.5)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.inf

    return max(
        directed_hausdorff(pred_pts, gt_pts)[0],
        directed_hausdorff(gt_pts, pred_pts)[0]
    )

def visualise_slice(img, mask_gt, mask_pred, save_path):
    mid = img.shape[2] // 2  # middle slice
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(img[:,:,mid], cmap="gray")
    plt.title("T2 Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(img[:,:,mid], cmap="gray")
    plt.imshow(mask_gt[:,:,mid], alpha=0.4)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(img[:,:,mid], cmap="gray")
    plt.imshow(mask_pred[:,:,mid], alpha=0.4, cmap="Reds")
    plt.title("Prediction")
    plt.axis("off")

    plt.savefig(save_path)
    plt.close()


# -------------------------
# Load holdout files
# -------------------------
holdout_imgs = sorted([f for f in os.listdir(FOLDER) if f.startswith("image_holdout")])
holdout_labs = sorted([f for f in os.listdir(FOLDER) if f.startswith("label_holdout")])

assert len(holdout_imgs) == len(holdout_labs), "Mismatch between images and labels."

# -------------------------
# Select 5 random patients
# -------------------------
random_ids = random.sample(range(len(holdout_imgs)), 5)
print(f"Evaluating patients:", random_ids)

# -------------------------
# Run evaluation
# -------------------------
for idx in random_ids:
    img = np.load(os.path.join(FOLDER, holdout_imgs[idx]))
    lab = np.load(os.path.join(FOLDER, holdout_labs[idx]))

    img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    if use_cuda:
        img_t = img_t.cuda()

    with torch.no_grad():
        pred = model(img_t).cpu().numpy()[0,0]

    dice = dice_score(pred, lab)
    hd   = hausdorff_distance(pred, lab)

    print(f"\nPatient {idx}:")
    print(f"  Dice score:      {dice:.4f}")
    print(f"  Hausdorff dist.: {hd:.2f}")

    # save visualisation
    save_path = f"./result/png/holdout_{idx:03d}.png"
    os.makedirs("./result/png", exist_ok=True)
    visualise_slice(img, lab, pred, save_path)
    print(f"  Saved visualisation → {save_path}")

print("\n✔ Holdout evaluation complete.")

