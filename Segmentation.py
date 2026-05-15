import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. LOAD IMAGES (same directory)
# ===============================
input_image = cv2.imread('data/Black_Footed_Albatross_0001_796111.jpg')
gt_image = cv2.imread('data/Black_Footed_Albatross_0001_796111.png', 0)

if input_image is None or gt_image is None:
    print("Error: Ensure 'input.jpg' and 'gt.png' are in same folder")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Ensure GT is binary
gt_image = (gt_image > 0).astype(np.uint8)

#plt.imshow(gray, cmap='gray')
# ===============================
# 2. OTSU SEGMENTATION
# ===============================
_, otsu_seg = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
#plt.imshow(otsu_seg, cmap='gray')


# ===============================
# 3. K-MEANS SEGMENTATION
# ===============================
pixel_vals = input_image.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
K = 3

_, labels, centers = cv2.kmeans(
    pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)
segmented = centers[labels.flatten()]
segmented = segmented.reshape(input_image.shape)

# Convert to grayscale mask
kmeans_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
_, kmeans_binary = cv2.threshold(
    kmeans_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# ===============================
# 4. AUTO POLARITY FIX (IMPORTANT)
# ===============================
def best_match(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    # IoU original
    inter1 = np.sum((pred == 1) & (gt == 1))
    union1 = np.sum((pred == 1) | (gt == 1))
    iou1 = inter1 / (union1 + 1e-8)

    # IoU inverted
    pred_inv = 1 - pred
    inter2 = np.sum((pred_inv == 1) & (gt == 1))
    union2 = np.sum((pred_inv == 1) | (gt == 1))
    iou2 = inter2 / (union2 + 1e-8)

    if iou2 > iou1:
        return pred_inv
    else:
        return pred

otsu_seg = best_match(otsu_seg, gt_image)
kmeans_binary = best_match(kmeans_binary, gt_image)

# ===============================
# 5. METRICS (NO SCIKIT)
# ===============================
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
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

    return accuracy, precision, recall, f1, iou, dice

otsu_metrics = compute_metrics(otsu_seg, gt_image)
kmeans_metrics = compute_metrics(kmeans_binary, gt_image)

# ===============================
# 6. PRINT RESULTS
# ===============================
def print_metrics(name, m):
    print(f"\n{name} Metrics:")
    print(f"Accuracy : {m[0]:.4f}")
    print(f"Precision: {m[1]:.4f}")
    print(f"Recall   : {m[2]:.4f}")
    print(f"F1 Score : {m[3]:.4f}")
    print(f"IoU      : {m[4]:.4f}")
    print(f"Dice     : {m[5]:.4f}")

print_metrics("Otsu", otsu_metrics)
print_metrics("K-Means", kmeans_metrics)

# ===============================
# 7. DISPLAY RESULTS
# ===============================
plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

plt.subplot(2,2,2)
plt.title("Ground Truth")
plt.imshow(gt_image, cmap='gray')

plt.subplot(2,2,3)
plt.title("Otsu (Final)")
plt.imshow(otsu_seg, cmap='gray')

plt.subplot(2,2,4)
plt.title("K-Means (Final)")
plt.imshow(kmeans_binary, cmap='gray')

plt.tight_layout()
plt.show()