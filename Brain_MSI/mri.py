import os
import csv
import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DATA_DIR = r"D:\Xử lý ảnh\Brai MIR\Patient-1"
FLAIR_FILE = os.path.join(DATA_DIR, "1-Flair.nii")
MASK_FILE = os.path.join(DATA_DIR, "1-LesionSeg-Flair.nii")
OUTPUT_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_title(tilte):
        print()
        print(tilte)

print_title("Kiem tra file")
if not os.path.isfile(FLAIR_FILE):
        raise FileNotFoundError(f"\nKhong tim thay file FLAIR:\n{FLAIR_FILE}")
if not os.path.isfile(MASK_FILE):
        raise FileNotFoundError(f"\nKhong tim thay file lesion:\n{MASK_FILE}")
print("FLAIR:")
print(FLAIR_FILE)
print("\nOutput:")
print(OUTPUT_DIR)

print_title("Doc du lieu nifti")
flair_img = nib.load(FLAIR_FILE)
mask_img = nib.load(MASK_FILE)
flair = flair_img.get_fdata()
mask = mask_img.get_fdata()
print("FLAIR shape:", flair.shape)
print("Mask shape :", mask.shape)

print("FLAIR dtype:", flair.dtype)
print("Mask dtype :", mask.dtype)

print("\nFLAIR intensity:")
print("Min :", flair.min())
print("Max :", flair.max())
print("Mean:", flair.mean())

print("\nMask:")
print("Min:", mask.min())
print("Max:", mask.max())

print("So voxel lesion:", np.count_nonzero(mask))

# Kiem tr kich thuoc
if flair.shape != mask.shape:
        raise ValueError("\nFLAIR va lesion mask khong cung kich thuoc")

# Tim slice co lesion nhieu nhat
print_title("Tim slice dai dien")
lesion_count = np.count_nonzero(mask > 0, axis = (0, 1))
slice_index = int(np.argmax(lesion_count))
print("Slice duoc chon:", slice_index)
print("So pixel lesion: ", lesion_count[slice_index])

# Lay slice
mri_slice = flair[:, :, slice_index]
mask_slice = mask[:, :, slice_index]
print("MRI slice:", mri_slice.shape)
print("Mask slice:", mask_slice.shape)

# Chuan hoa MRI ve 0-255
print_title("Chuan hoa cuong do anh")

def normalize_image(image):
        image = image.astype(np.float32)
        non_zero = image[image > 0]
        if len(non_zero) == 0:
                raise ValueError("Anh khong co picel khac 0")
        p1 = np.percentile(non_zero, 1)
        p99 = np.percentile(non_zero, 99)
        print("Percentile 1%: ", p1)
        print("Percentile 99%: ", p99)
        if p99 <= p1:
                p1 = non_zero.min()
                p99 = non_zero.max()
        normalized = (image - p1) / (p99 - p1)
        normalized = np.clip(normalized, 0, 1)
        normalized = (normalized * 255).astype(np.uint8)
        return normalized
mri_8bit = normalize_image(mri_slice)
print("Normalized min:", mri_8bit.min())
print("Normalized max:", mri_8bit.max())

# Lesion mask nhi phan
lesion_mask = (mask_slice > 0)
print_title("LESION MASK")
print("So pixel lesion:", np.count_nonzero(lesion_mask))

# Tao roi lesion
roi_lesion = lesion_mask.copy()

# Hien thi mir va huong dan
print_title("Chon ROI bang chuot")

# Ham chon ROI
def select_roi(image, title):
        print()
        print(title)
        roi = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        x, y, w, h = roi
        if w == 0 or h == 0:
                raise ValueError(f"\nROI khong hop le cho {title}")
        return(int(x), int(y), int(w), int(h))

# Hien thi anh grayscal
roi_background_box = select_roi(mri_8bit, "ROI 1 - BACKGROUND - Keo chuot, ENTER de xac nhan")
roi_brain_box = select_roi(mri_8bit, "ROI 2 - NORMAL BRAIN - Keo chuot, ENTER de xac nhan")

# Tao mask cho roi 1
x1, y1, w1, h1 = (roi_background_box)
roi_background_mask = (np.zeros(mri_8bit.shape, dtype=bool))
roi_background_mask[y1: y1+h1, x1:x1+w1]=True

# Tao mask cho roi 2
x2, y2, w2, h2 = (roi_brain_box)
roi_brain_mask = (np.zeros(mri_8bit.shape, dtype=bool))
roi_brain_mask[y2:y2+h2, x2:x2+w2]=True

# Loai lesion khoi roi normal brain
roi_brain_mask = (roi_brain_mask & (~lesion_mask))

# Kiem tra roi
print_title("Kiem tra 3 ROI")
print("ROI 1 Background:", np.count_nonzero(roi_background_mask), "pixels")
print("ROI 2 Normal brain:", np.count_nonzero(roi_brain_mask), "pixels")
print("ROI 3 Lesion:", np.count_nonzero(roi_lesion), "pixels")

# Lay pixels cua 3 roi
background_pixels = mri_slice[roi_background_mask]
brain_pixels = mri_slice[roi_brain_mask]
lesion_pixels = mri_slice[roi_lesion]

# Ham tinh thong ke roi
def calculate_statistics(name, pixels):
        if len(pixels) == 0:
                print(f"{name}: Khong co pixel.")
                return{"ROI": name, "PixelCount": 0, "Mean": np.nan, "Std": np.nan, "Min": np.nan, "Max": np.nan}
        mean_value = np.mean(pixels)
        std_value = np.std(pixels)
        min_value = np.min(pixels)
        max_value = np.max(pixels)
        print()
        print(name)
        print("Pixel count:", len(pixels))
        print("Mean:", mean_value)
        print("Standard deviation:", std_value)
        print("Min:", min_value)
        print("Max:", max_value)
        return {"ROI": name, "PixelCount": len(pixels), "Mean": mean_value, "Std": std_value, "Min": min_value, "Max": max_value}

# Tinh Statistics
print_title("Thong ke 3 ROI")
statistics = []
statistics.append(calculate_statistics("ROI 1 - Background", background_pixels))
statistics.append(
calculate_statistics("ROI 2 - Normal Brain", brain_pixels))
statistics.append(
calculate_statistics("ROI 3 - Lesion", lesion_pixels))

# Luu ket qua ROI ra CSV
csv_file = os.path.join(OUTPUT_DIR, "roi_statistics.csv")
with open(csv_file, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["ROI", "PixelCount", "Mean", "Std", "Min", "Max"])
        writer.writeheader()
        writer.writerows(statistics)
print("\nĐã lưu:", csv_file)

# Ve 3 ROI tren mri
print_title("Ve 3 ROI")
plt.figure(figsize=(8, 8))
plt.imshow(mri_8bit, cmap="gray")

# ROI 1
rect1 = Rectangle((x1, y1), w1, h1, fill=False, edgecolor="red", linewidth=2)
plt.gca().add_patch(rect1)
plt.text(x1, y1 - 5, "ROI 1 - Background", color="red", fontsize=10, weight="bold")

# ROI 2
rect2 = Rectangle((x2, y2), w2, h2, fill=False, edgecolor="lime", linewidth=2)
plt.gca().add_patch(rect2)
plt.text(x2, y2 - 5, "ROI 2 - Normal Brain", color = "lime", fontsize=10, weight="bold")

# ROI 3
masked_lesion = np.ma.masked_where(~roi_lesion, roi_lesion)
plt.imshow(masked_lesion, cmap="autumn", alpha=0.6)
plt.title(f"Three ROIs - FLAIR Slice {slice_index}")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_three_ROIs.png"),dpi=300, bbox_inches="tight")
plt.show()

# Histogram cua 3 ROI
print_title("HISTOGRAM 3 ROI")
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(background_pixels, bins=50)
plt.title("ROI 1 - Background")
plt.xlabel("MRI intensity")
plt.ylabel("Pixel count")
plt.subplot(2, 2, 2)
plt.hist(brain_pixels, bins=50)
plt.title("ROI 2 - Normal Brain")
plt.xlabel("MRI intensity")
plt.ylabel("Pixel count")
plt.subplot(2, 2, 3)
plt.hist(lesion_pixels, bins=50)
plt.title("ROI 3 - Lesion")
plt.xlabel("MRI intensity")
plt.ylabel("Pixel count")
plt.subplot(2, 2, 4)
plt.hist(background_pixels, bins=50, alpha=0.5, label="Background")
plt.hist(brain_pixels, bins=50, alpha=0.5, label="Normal Brain")
plt.hist(lesion_pixels, bins=50, alpha=0.5, label="Lesion")
plt.title("So sanh Histogram 3 ROI")
plt.xlabel("MRI intensity")
plt.ylabel("Pixel count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_ROI_histograms.png"), dpi=300, bbox_inches="tight")
plt.show()

# Gaussian Filter
print_title("Gaussian Filter")
gaussian_3 = cv2.GaussianBlur(mri_8bit, (3, 3), 0)
gaussian_7 = cv2.GaussianBlur(mri_8bit, (7, 7), 0)
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(mri_8bit, cmap="gray")
plt.title("Original")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(gaussian_3, cmap="gray")
plt.title("Gaussian  3x3")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(gaussian_7, cmap="gray")
plt.title("Gaussian 7x7")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_gaussian_filter.png"), dpi=300, bbox_inches="tight")
plt.show()

# Median filter
print_title("MEDIAN FILTER")
median_3 = cv2.medianBlur(mri_8bit, 3)
median_7 = cv2.medianBlur(mri_8bit, 7)
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(mri_8bit, cmap="gray")
plt.title("Original")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(median_3, cmap="gray")
plt.title("Median 3x3")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(median_7,cmap="gray")
plt.title("Median 7x7")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_median_filter.png"), dpi=300, bbox_inches="tight")
plt.show()

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(mri_8bit)
print_title("CLAHE")
plt.subplot(1, 2, 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(mri_8bit, cmap="gray")
plt.title("FLAIR trước CLAHE")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(clahe_image, cmap="gray")
plt.title("FLAIR sau CLAHE")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_clahe_comparison.png"), dpi=300, bbox_inches="tight")
plt.show()

# Histogram truoc va sau CLAHE
print_title("HISTOGRAM CLAHE")

# Choi lay vung tin hieu MRI
brain_area = (mri_slice > 0)
hist_before = mri_8bit[brain_area]
hist_after = clahe_image[brain_area]
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(hist_before, bins=100, range=(0, 255))
plt.title("Histogram truoc CLAHE")
plt.xlabel("Intensity")
plt.ylabel("Pixel count")
plt.subplot(1, 2, 2)
plt.hist(hist_after, bins=100, range=(0, 255))
plt.title("Histogram sau CLAHE")
plt.xlabel("Intensity")
plt.ylabel("Pixel count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_clahe_histogram.png"), dpi=300, bbox_inches="tight")
plt.show()

# Luu mask lesion
mask_png = (lesion_mask.astype(np.uint8) * 255)
cv2.imwrite(os.path.join(OUTPUT_DIR, "07_lesion_mask.png"), mask_png)

# Luu anh clahe
cv2.imwrite(os.path.join(OUTPUT_DIR, "08_flair_clahe.png"), clahe_image)

# Tong ket
print_title("HOÀN THÀNH XỬ LÝ MRI")
print("Patient:", os.path.basename(DATA_DIR))
print("Slice:", slice_index)
print("ROI 1 - Background:", len(background_pixels), "pixels")
print("ROI 2 - Normal Brain:", len(brain_pixels), "pixels")
print("ROI 3 - Lesion:", len(lesion_pixels), "pixels")
print("\nCác file kết quả:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(" -", filename)
        print("\nKết quả được lưu tại:")
        print(OUTPUT_DIR)
        print("\nChương trình kết thúc.")
