import numpy as np
from scipy.ndimage import label

# =========================================================
# PATHS — EDIT IF NEEDED
# =========================================================
pred_dir = "/cs/student/projects4/misc/alukic/MPHY0041/tutorials/segmentation/result_lesion_roi/"
gt_dir   = "/cs/student/projects4/misc/alukic/outputs/lesion-data/"

threshold = 0.35

# =========================================================
# HELPER FUNCTION
# =========================================================
def compute_fp_for_case(pred, gt, thr):
    pred_bin = (pred >= thr).astype(np.uint8)

    # label connected components
    labeled, num = label(pred_bin)

    false_positives = 0

    for i in range(1, num + 1):
        comp = (labeled == i)

        # check overlap with ground truth
        overlap = np.logical_and(comp, gt > 0).sum()

        if overlap == 0:
            false_positives += 1

    return false_positives, num


# =========================================================
# LOOP THROUGH CASES
# =========================================================
results = []

for case_id in range(100):  # adjust range if needed
    try:
        pred_path = f"{pred_dir}/pred_step1200_id{case_id:03d}.npy"
        gt_path   = f"{gt_dir}/label_test{case_id:03d}.npy"

        pred = np.load(pred_path)
        gt   = np.load(gt_path)

        fp, total_pred = compute_fp_for_case(pred, gt, threshold)

        results.append(fp)

        print(f"Case {case_id:03d} → FP: {fp}, total predicted regions: {total_pred}")

    except:
        continue

# =========================================================
# SUMMARY
# =========================================================
results = np.array(results)

print("\n========== SUMMARY ==========")
print(f"Mean FP per case: {results.mean():.2f}")
print(f"Median FP per case: {np.median(results):.2f}")
print(f"Max FP in a case: {results.max()}")
