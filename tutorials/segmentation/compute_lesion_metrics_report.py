import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import label

# ============================================================
# PATHS
# ============================================================

PRED_DIR = "./result_lesion_roi"
GT_DIR   = "/cs/student/projects4/misc/alukic/outputs/lesion-data"

# only use the final checkpoint predictions
PRED_GLOB = os.path.join(PRED_DIR, "pred_step1200_id*.npy")
GT_GLOB   = os.path.join(GT_DIR, "label_train*.npy")

OUT_CSV_PER_CASE = "lesion_metrics_per_case.csv"
OUT_CSV_SUMMARY  = "lesion_metrics_summary.csv"
OUT_FIGURE       = "lesion_metrics_report_figure.png"

# post-processing
THR = 0.35
MIN_VOX = 1

# ============================================================
# HELPERS
# ============================================================

def extract_case_id(path):
    name = os.path.basename(path)

    # prediction: pred_step1200_id028.npy
    m = re.search(r'id(\d{3})', name)
    if m:
        return m.group(1)

    # GT: label_train028.npy
    m = re.search(r'label_train(\d{3})', name)
    if m:
        return m.group(1)

    return None


def keep_components_min_size(binary_mask, min_vox=1):
    labeled, num = label(binary_mask)
    out = np.zeros_like(binary_mask, dtype=np.uint8)

    kept = 0
    for i in range(1, num + 1):
        comp = (labeled == i)
        if comp.sum() >= min_vox:
            out[comp] = 1
            kept += 1

    return out, kept


def lesion_level_sensitivity(gt_mask, pred_mask):
    gt_labeled, num_gt = label(gt_mask)

    if num_gt == 0:
        return np.nan, 0, 0

    tp_lesions = 0
    for i in range(1, num_gt + 1):
        gt_component = (gt_labeled == i)
        if np.any(pred_mask[gt_component] > 0):
            tp_lesions += 1

    return tp_lesions / num_gt, tp_lesions, num_gt


def false_positives_per_case(gt_mask, pred_mask):
    pred_labeled, num_pred = label(pred_mask)

    if num_pred == 0:
        return 0, 0, 0

    fp_count = 0
    tp_pred_regions = 0

    for i in range(1, num_pred + 1):
        pred_component = (pred_labeled == i)
        if np.any(gt_mask[pred_component] > 0):
            tp_pred_regions += 1
        else:
            fp_count += 1

    return fp_count, num_pred, tp_pred_regions


# ============================================================
# FIND FILES
# ============================================================

pred_files = sorted(glob.glob(PRED_GLOB))
gt_files = sorted(glob.glob(GT_GLOB))

print(f"Found {len(pred_files)} prediction files")
print(f"Found {len(gt_files)} GT files")

if len(pred_files) == 0:
    raise RuntimeError(f"No prediction files found with pattern: {PRED_GLOB}")

if len(gt_files) == 0:
    raise RuntimeError(f"No GT files found with pattern: {GT_GLOB}")

pred_map = {}
for f in pred_files:
    cid = extract_case_id(f)
    if cid is not None:
        pred_map[cid] = f

gt_map = {}
for f in gt_files:
    cid = extract_case_id(f)
    if cid is not None:
        gt_map[cid] = f

common_case_ids = sorted(set(pred_map.keys()) & set(gt_map.keys()))
print(f"Matched {len(common_case_ids)} cases")

if len(common_case_ids) == 0:
    print("Example prediction files:")
    for f in pred_files[:5]:
        print("  ", os.path.basename(f))
    print("Example GT files:")
    for f in gt_files[:5]:
        print("  ", os.path.basename(f))
    raise RuntimeError("No matching case IDs found. Check filename patterns.")

# ============================================================
# METRICS
# ============================================================

rows = []

for case_id in common_case_ids:
    pred_path = pred_map[case_id]
    gt_path = gt_map[case_id]

    pred_prob = np.load(pred_path)
    gt_mask = (np.load(gt_path) > 0).astype(np.uint8)

    if pred_prob.shape != gt_mask.shape:
        print(f"Skipping case {case_id}: shape mismatch {pred_prob.shape} vs {gt_mask.shape}")
        continue

    pred_mask = (pred_prob >= THR).astype(np.uint8)
    pred_mask, kept_regions = keep_components_min_size(pred_mask, min_vox=MIN_VOX)

    sens, tp_lesions, total_gt_lesions = lesion_level_sensitivity(gt_mask, pred_mask)
    fp_count, total_pred_regions, tp_pred_regions = false_positives_per_case(gt_mask, pred_mask)

    fp_percent = 100.0 * fp_count / total_pred_regions if total_pred_regions > 0 else 0.0
    precision_percent = 100.0 * tp_pred_regions / total_pred_regions if total_pred_regions > 0 else 0.0

    rows.append({
        "case_id": case_id,
        "total_gt_lesions": total_gt_lesions,
        "tp_lesions_detected": tp_lesions,
        "lesion_sensitivity": sens,
        "total_predicted_regions": total_pred_regions,
        "tp_predicted_regions": tp_pred_regions,
        "fp_count": fp_count,
        "fp_percent": fp_percent,
        "precision_percent": precision_percent,
    })

    sens_str = "nan" if np.isnan(sens) else f"{sens:.3f}"
    print(
        f"Case {case_id} | GT lesions: {total_gt_lesions} | "
        f"Detected: {tp_lesions} | Sensitivity: {sens_str} | "
        f"FP: {fp_count} | Pred regions: {total_pred_regions} | "
        f"FP%: {fp_percent:.1f}"
    )

df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No valid matched cases remained after shape checks.")

# ============================================================
# SAVE PER-CASE CSV
# ============================================================

df_sorted = df.sort_values(by=["fp_count", "lesion_sensitivity"], ascending=[False, True])
df_sorted.to_csv(OUT_CSV_PER_CASE, index=False)
print(f"\nSaved per-case CSV: {OUT_CSV_PER_CASE}")

# ============================================================
# SUMMARY
# ============================================================

valid_sens = df["lesion_sensitivity"].dropna()

summary = pd.DataFrame({
    "metric": [
        "Number of matched cases",
        "Mean lesion sensitivity",
        "Median lesion sensitivity",
        "Min lesion sensitivity",
        "Max lesion sensitivity",
        "Mean FP per case",
        "Median FP per case",
        "Max FP per case",
        "Mean FP percentage",
        "Median FP percentage",
        "Mean precision percentage",
        "Median precision percentage",
    ],
    "value": [
        len(df),
        round(valid_sens.mean(), 4) if len(valid_sens) else np.nan,
        round(valid_sens.median(), 4) if len(valid_sens) else np.nan,
        round(valid_sens.min(), 4) if len(valid_sens) else np.nan,
        round(valid_sens.max(), 4) if len(valid_sens) else np.nan,
        round(df["fp_count"].mean(), 4),
        round(df["fp_count"].median(), 4),
        int(df["fp_count"].max()),
        round(df["fp_percent"].mean(), 4),
        round(df["fp_percent"].median(), 4),
        round(df["precision_percent"].mean(), 4),
        round(df["precision_percent"].median(), 4),
    ]
})

summary.to_csv(OUT_CSV_SUMMARY, index=False)
print(f"Saved summary CSV: {OUT_CSV_SUMMARY}")

print("\n===== SUMMARY =====")
print(summary.to_string(index=False))

# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(figsize=(12, 4.8))

# Panel A: lesion sensitivity histogram
ax1 = plt.subplot(1, 2, 1)
ax1.hist(valid_sens, bins=10, edgecolor="black")
ax1.axvline(valid_sens.mean(), linestyle="--", linewidth=1.5,
            label=f"Mean = {valid_sens.mean():.2f}")
ax1.axvline(valid_sens.median(), linestyle=":", linewidth=1.5,
            label=f"Median = {valid_sens.median():.2f}")
ax1.set_xlabel("Lesion-level sensitivity per case")
ax1.set_ylabel("Number of cases")
ax1.set_title("A. Distribution of lesion-level sensitivity")
ax1.legend(frameon=False)

# Panel B: FP per case boxplot
ax2 = plt.subplot(1, 2, 2)
ax2.boxplot(df["fp_count"], vert=True)
ax2.set_ylabel("False positives per case")
ax2.set_xticks([1])
ax2.set_xticklabels(["All cases"])
ax2.set_title("B. False positives per case")

plt.tight_layout()
plt.savefig(OUT_FIGURE, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure: {OUT_FIGURE}")

# ============================================================
# TOP 10 WORST CASES
# ============================================================

print("\n===== TOP 10 WORST CASES BY FP COUNT =====")
print(
    df.sort_values(by="fp_count", ascending=False)
      .head(10)[[
          "case_id",
          "total_gt_lesions",
          "tp_lesions_detected",
          "lesion_sensitivity",
          "total_predicted_regions",
          "fp_count",
          "fp_percent"
      ]]
      .to_string(index=False)
)
