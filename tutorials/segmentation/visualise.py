# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
# run train.py before visualise the results
import os
import numpy as np
import matplotlib.pyplot as plt

# paths to your data and results
path_to_data = './data/promise12-data'
path_to_save = './result'

# plot example slices of segmentation results
for ext in ["-tf.npy","-pt.npy"]:  # find all npy files
    files = [f for f in os.listdir(path_to_save) if f.endswith(ext)]  
    if not files:
        print(f"No files found for extension {ext}, skipping.")
        continue

    fmax = []  # find the maximum step for each test ID
    for test_id in set([f.split('_')[1] for f in files]):
        fmax += [max([f for f in files if f.split('_')[1] == test_id])]

    for f in fmax:
        # load prediction and corresponding input image
        label = np.load(os.path.join(path_to_save, f))
        image_file = os.path.join(path_to_data, "image_" + f.split('_')[1] + ".npy")
        if not os.path.exists(image_file):
            print(f"Image file {image_file} not found, skipping.")
            continue
        image = np.load(image_file)[::2, ::2, ::2]  # subsample as per loader

        # select subset of slices for visualization
        slices = range(0, label.shape[0], 3)  # only display every 3rd slice
        montage = np.concatenate([
            np.concatenate([image[i, ...] for i in slices], axis=0),
            np.concatenate([label[i, ...] * np.max(image) for i in slices], axis=0)
        ], axis=1)

        # save montage as PNG
        filepath_to_save = os.path.join(path_to_save, f.split('.')[0] + '.png')
        plt.imsave(filepath_to_save, montage, cmap='gray')
        print(f'Plot saved: {filepath_to_save}')

print('All plots saved in:', path_to_save)
