import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# IMAGE AND MASK FOLDERS
# ==========================================

image_folder = "images"
mask_folder = "masks"

image_files = sorted(os.listdir(image_folder))

# ==========================================
# METRICS FUNCTION
# ==========================================

def compute_metrics(pred, gt):

    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    precision = TP / (TP + FP + 1e-8)

    recall = TP / (TP + FN + 1e-8)

    f1 = 2 * precision * recall / (
        precision + recall + 1e-8
    )

    iou = TP / (TP + FP + FN + 1e-8)

    dice = (2 * TP) / (
        2 * TP + FP + FN + 1e-8
    )

    return accuracy, precision, recall, f1, iou, dice

# ==========================================
# POLARITY FIX
# ==========================================

def best_match(pred, gt):

    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    # Original IoU
    inter1 = np.sum((pred == 1) & (gt == 1))
    union1 = np.sum((pred == 1) | (gt == 1))
    iou1 = inter1 / (union1 + 1e-8)

    # Inverted IoU
    pred_inv = 1 - pred

    inter2 = np.sum((pred_inv == 1) & (gt == 1))
    union2 = np.sum((pred_inv == 1) | (gt == 1))
    iou2 = inter2 / (union2 + 1e-8)

    if iou2 > iou1:
        return pred_inv
    else:
        return pred

# ==========================================
# STORE ALL RESULTS
# ==========================================

otsu_results = []
kmeans_results = []

# ==========================================
# LOOP THROUGH ALL IMAGES
# ==========================================

for file in image_files[:5]:

    image_path = os.path.join(
        image_folder,
        file
    )

    # Mask path
    mask_name = file.replace(".jpg", ".png")

    mask_path = os.path.join(
        mask_folder,
        mask_name
    )

    # Read images
    image = cv2.imread(image_path)

    gt = cv2.imread(mask_path, 0)

    if image is None or gt is None:
        print("Skipping:", file)
        continue

    # Convert GT to binary
    gt = (gt > 0).astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ======================================
    # OTSU SEGMENTATION
    # ======================================

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ======================================
    # KMEANS SEGMENTATION
    # ======================================

    pixel_vals = image.reshape((-1, 3))

    pixel_vals = np.float32(pixel_vals)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    K = 3

    _, labels, centers = cv2.kmeans(
        pixel_vals,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)

    segmented = centers[
        labels.flatten()
    ]

    segmented = segmented.reshape(
        image.shape
    )

    # Convert to binary
    kmeans_gray = cv2.cvtColor(
        segmented,
        cv2.COLOR_BGR2GRAY
    )

    _, kmeans_binary = cv2.threshold(
        kmeans_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ======================================
    # POLARITY FIX
    # ======================================

    otsu = best_match(otsu, gt)

    kmeans_binary = best_match(
        kmeans_binary,
        gt
    )

    # ======================================
    # COMPUTE METRICS
    # ======================================

    otsu_metric = compute_metrics(
        otsu,
        gt
    )

    kmeans_metric = compute_metrics(
        kmeans_binary,
        gt
    )

    otsu_results.append(otsu_metric)

    kmeans_results.append(kmeans_metric)

    print(f"\nProcessed: {file}")

# ==========================================
# AVERAGE RESULTS
# ==========================================

otsu_avg = np.mean(
    otsu_results,
    axis=0
)

kmeans_avg = np.mean(
    kmeans_results,
    axis=0
)

# ==========================================
# PRINT FINAL RESULTS
# ==========================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "IoU",
    "Dice"
]

print("\n========== OTSU AVERAGE ==========")

for i in range(len(metric_names)):

    print(
        metric_names[i],
        ":",
        round(otsu_avg[i], 4)
    )

print("\n========== KMEANS AVERAGE ==========")

for i in range(len(metric_names)):

    print(
        metric_names[i],
        ":",
        round(kmeans_avg[i], 4)
    )