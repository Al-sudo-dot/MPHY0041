import os, json
import numpy as np
import torch
from torch.utils.data import Dataset


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crop_patch(img_czyx, lab_zyx, center_zyx, pz, py, px):
    C, Z, Y, X = img_czyx.shape
    hz, hy, hx = pz // 2, py // 2, px // 2
    cz, cy, cx = map(int, center_zyx)

    cz = clamp(cz, hz, Z - hz)
    cy = clamp(cy, hy, Y - hy)
    cx = clamp(cx, hx, X - hx)

    z0, z1 = cz - hz, cz + hz
    y0, y1 = cy - hy, cy + hy
    x0, x1 = cx - hx, cx + hx

    img_p = img_czyx[:, z0:z1, y0:y1, x0:x1]
    lab_p = lab_zyx[z0:z1, y0:y1, x0:x1]
    return img_p, lab_p


def roi_box_to_zyx(roi_box):
    """
    Accepts either:
      [z0, z1, y0, y1, x0, x1]
    or dict with keys (z0,z1,y0,y1,x0,x1)
    """
    if isinstance(roi_box, dict):
        return int(roi_box["z0"]), int(roi_box["z1"]), int(roi_box["y0"]), int(roi_box["y1"]), int(roi_box["x0"]), int(roi_box["x1"])
    return tuple(int(v) for v in roi_box)


class LesionPatchDatasetROI(Dataset):
    """
    Loads:
      image_{split}{id:03d}.npy  (C,Z,Y,X)
      label_{split}{id:03d}.npy  (Z,Y,X) or (1,Z,Y,X)

    Samples patches inside prostate ROI using roi_json with per-ID boxes.

    Patch sampling:
      - pos_fraction: choose lesion-centered patches if lesion exists
      - otherwise: random inside ROI box

    Returns:
      x: (C,pz,py,px) float32
      y: (1,pz,py,px) float32 binary
    """
    def __init__(
        self,
        data_dir,
        roi_json,
        split="train",
        ids=None,
        pz=32, py=96, px=96,
        patches_per_volume=32,
        pos_fraction=0.5,
        seed=0,
    ):
        self.data_dir = data_dir
        self.split = split
        self.pz, self.py, self.px = pz, py, px
        self.patches_per_volume = int(patches_per_volume)
        self.pos_fraction = float(pos_fraction)
        self.rng = np.random.default_rng(seed)

        with open(roi_json, "r") as f:
            self.rois = json.load(f)

        if ids is None:
            prefix = f"image_{split}"
            found = []
            for fn in os.listdir(data_dir):
                if fn.startswith(prefix) and fn.endswith(".npy"):
                    num = fn.replace(prefix, "").replace(".npy", "")
                    found.append(int(num))
            self.ids = sorted(found)
        else:
            self.ids = list(ids)

        self.index = []
        npos = int(round(self.patches_per_volume * self.pos_fraction))
        for vid in self.ids:
            for k in range(self.patches_per_volume):
                is_pos = (k < npos)
                self.index.append((vid, is_pos))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        vid, is_pos = self.index[i]

        img_path = os.path.join(self.data_dir, f"image_{self.split}{vid:03d}.npy")
        lab_path = os.path.join(self.data_dir, f"label_{self.split}{vid:03d}.npy")

        img = np.load(img_path).astype(np.float32)  # (C,Z,Y,X)
        lab = np.load(lab_path)
        if lab.ndim == 4:
            lab = lab[0]
        lab = (lab > 0).astype(np.uint8)            # (Z,Y,X)

        C, Z, Y, X = img.shape

        # ROI box for this id: keys might be "000", 0, or "test000" depending on how you saved it.
        key_candidates = [f"{vid:03d}", str(vid), f"test{vid:03d}", f"train{vid:03d}", f"holdout{vid:03d}"]
        roi_box = None
        for k in key_candidates:
            if k in self.rois:
                roi_box = self.rois[k]
                break
        if roi_box is None:
            # last resort: whole volume
            z0, z1, y0, y1, x0, x1 = 0, Z, 0, Y, 0, X
        else:
            z0, z1, y0, y1, x0, x1 = roi_box_to_zyx(roi_box)
            # clamp to bounds
            z0, z1 = clamp(z0, 0, Z), clamp(z1, 0, Z)
            y0, y1 = clamp(y0, 0, Y), clamp(y1, 0, Y)
            x0, x1 = clamp(x0, 0, X), clamp(x1, 0, X)

        # sample center
        if is_pos and lab.sum() > 0:
            coords = np.argwhere(lab > 0)
            cz, cy, cx = coords[self.rng.integers(0, coords.shape[0])]
        else:
            cz = self.rng.integers(z0, max(z0 + 1, z1))
            cy = self.rng.integers(y0, max(y0 + 1, y1))
            cx = self.rng.integers(x0, max(x0 + 1, x1))

        img_p, lab_p = crop_patch(img, lab, (cz, cy, cx), self.pz, self.py, self.px)

        x = torch.from_numpy(img_p)                 # (C,pz,py,px)
        y = torch.from_numpy(lab_p[None]).float()   # (1,pz,py,px)
        return x, y
