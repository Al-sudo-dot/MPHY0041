import numpy as np
from scipy.ndimage import label

def postprocess_mask(mask, min_vox=0, keep_largest=False):
    """
    mask: binary 3D numpy array
    min_vox: remove connected components smaller than this
    keep_largest: keep only the largest connected component
    """
    mask = mask.astype(bool)

    labeled, num = label(mask)
    if num == 0:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background

    out = np.zeros_like(mask)

    if keep_largest:
        largest = sizes.argmax()
        out = labeled == largest
    else:
        for i in range(1, len(sizes)):
            if sizes[i] >= min_vox:
                out[labeled == i] = True

    return out
