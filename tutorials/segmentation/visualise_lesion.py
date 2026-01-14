import os
import numpy as np
import matplotlib.pyplot as plt

DATA = "./data/lesion-data"
PRED = "./result_lesion"
OUT  = "./result_lesion/png"
os.makedirs(OUT, exist_ok=True)

def load_test_image(idx):
    path = os.path.join(DATA, f"image_test{idx:03d}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing test image: {path}")
    return np.load(path)

def extract_ids(filename):
    name = filename.replace(".npy", "")

    if "_id" in name:
        step = int(name.split("_")[1].replace("step", ""))
        pid  = int(name.split("_")[2].replace("id", ""))
        return step, pid
    else:
        step = int(name.split("_")[1].replace("step", ""))
        pid  = 0
        return step, pid

def plot_strip(img, pred, savepath, title):
    nz = img.shape[2]
    stepz = max(1, nz // 8)
    slices = list(range(0, nz, stepz))[:8]

    fig, axarr = plt.subplots(len(slices), 2, figsize=(6, 16))
    fig.suptitle(title)

    for i, z in enumerate(slices):
        axarr[i, 0].imshow(img[:, :, z], cmap="gray")
        axarr[i, 0].set_title(f"Image z={z}")
        axarr[i, 0].axis("off")

        axarr[i, 1].imshow(pred[:, :, z], cmap="gray")
        axarr[i, 1].set_title("Prediction")
        axarr[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(savepath, dpi=120)
    plt.close()
    print(f"✓ Saved strip: {savepath}")

pred_files = sorted([f for f in os.listdir(PRED) if f.endswith(".npy")])
print(f"Found {len(pred_files)} prediction files.")

for f in pred_files:
    if not f.startswith("pred_"):
        continue  # avoids grabbing model weights or other npys

    print(f"Processing: {f}")
    step, pid = extract_ids(f)

    pred = np.load(os.path.join(PRED, f))
    img  = load_test_image(pid)

    savepath = os.path.join(OUT, f"lesion_{pid:03d}_step{step:04d}.png")
    plot_strip(img, pred, savepath, f"Lesion ID {pid} step {step}")
