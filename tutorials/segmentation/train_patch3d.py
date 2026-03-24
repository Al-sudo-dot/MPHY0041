import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_patches3d_roi import LesionPatchDatasetROI
from model_unet3d_small import UNet3DSmall


def soft_dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum()
    denom = probs.sum() + target.sum() + eps
    return 1.0 - (2.0 * inter / denom)


def hard_dice(logits, target, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum() + eps
    return (2.0 * inter / denom).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--roi_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--pz", type=int, default=32)
    ap.add_argument("--py", type=int, default=96)
    ap.add_argument("--px", type=int, default=96)

    ap.add_argument("--patches_per_vol", type=int, default=32)
    ap.add_argument("--pos_frac", type=float, default=0.5)

    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--bce_w", type=float, default=0.3)
    ap.add_argument("--dice_w", type=float, default=0.7)

    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    train_ds = LesionPatchDatasetROI(
        data_dir=args.data_dir,
        roi_json=args.roi_json,
        split="train",
        pz=args.pz, py=args.py, px=args.px,
        patches_per_volume=args.patches_per_vol,
        pos_fraction=args.pos_frac,
        seed=args.seed
    )
    val_ds = LesionPatchDatasetROI(
        data_dir=args.data_dir,
        roi_json=args.roi_json,
        split="holdout",
        pz=args.pz, py=args.py, px=args.px,
        patches_per_volume=max(8, args.patches_per_vol // 4),
        pos_fraction=args.pos_frac,
        seed=args.seed + 1
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = UNet3DSmall(in_ch=3, base=16, out_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_it = iter(train_loader)
    best_val = -1.0

    for step in range(1, args.steps + 1):
        try:
            x, y = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            x, y = next(train_it)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.train()
        logits = model(x)

        bce = F.binary_cross_entropy_with_logits(logits, y)
        dls = soft_dice_loss(logits, y)
        loss = args.bce_w * bce + args.dice_w * dls

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            d = hard_dice(logits.detach(), y)
            print(f"[step {step:05d}] loss={loss.item():.4f}  bce={bce.item():.4f}  dice={d:.3f}")

        if step % args.save_every == 0:
            model.eval()
            dices = []
            with torch.no_grad():
                for j, (vx, vy) in enumerate(val_loader):
                    vx = vx.to(device, non_blocking=True)
                    vy = vy.to(device, non_blocking=True)
                    vlg = model(vx)
                    dices.append(hard_dice(vlg, vy))
                    if j >= 20:
                        break
            mean_val = float(np.mean(dices)) if dices else 0.0
            print(f"[val] step {step:05d} mean_dice={mean_val:.4f}")

            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
                "val_dice": mean_val,
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_step{step:05d}.pth"))

            if mean_val > best_val:
                best_val = mean_val
                torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
                print(f"[saved] new best {best_val:.4f}")

    print("[done] training complete")


if __name__ == "__main__":
    main()
