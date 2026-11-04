import cv2
import numpy as np
import matplotlib.pyplot as plt
from roboflow import KMeans

SCALE_BAR_UM = 50 # Chinh theo anh

# Model

rf = Roboflow(api_key="_____")
project = rf.workspace().project("______")
model = project.version("___").model

# Load Model

image_path = r"______"
image = cv2.imread(image_path)

if image is None:
        print("Khong load duoc anh!!!!")
        exit()

# SCALE BAR
def dectect_scale_bar(image):
        gray = cv2.cvtClor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > 0 and w/h > 5 and w > 50:
                        candidates.append((x, y, w, h))

        return max(candidates, key=lambda x : x[2]) if candidates else None
scale = dectect_scale_bar(image)
if scale:
        x, y, w, h = scale
        scale_bar_pixels = w
else :
        print("Fallback scale")
        scale_bar_pixels = 200
um_per_pixel = SCALE_BAR_UM / scale_bar_pixels
are_convert = um_per_pixel ** 2
# PREPROCESS
def enhance(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(2.0, (8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def erosion(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.erode(gray, np.ones((5, 5), np.uint8), 1)

def kmeans2(image):
        pixels = image.reshape(-1, 1).astype(np.float32)
        kmeans = KMeans(n_clusters=2, n_init=10).fit(pixels)
        return kmeans.labels_.reshape(image.shape)

# PRESICT
preds = model.predict(image_path, confidence=40, overlap=30).json()

#MAIN
for i,  pred in enumerate(preds['predictions']):
        print(f"\nREGION {i+1}")

        #mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = np.array([[int(p['x']), int(p['y'])] for p in pred['points']], np.int32)
        cv2.fillPoly(mask, [pts], 255)

        seg = cv2.bitwise_and(image, image, mask=mask)

        #crop
        x, y, w, h = map(int, [pred['x'], pred['y'], pred['width'], pred['height']])
        x1, y1 = max(0, x-w//2), max(0, y-h//2)
        x2, y2 = min(image.shape[1], x+w//2), min(image.shape[0], y+h//2)

        crop = seg[y1:y2, x1:x2]
        if crop.size == 0:
                continue

        # enhance
        enhanced = enhance(crop)

        # erosion
        eroded = erosion(enhanced)

        # kmeans nucleus
        labels = kmeans2(eroded)

        # Chon nucleus theo do toi
        if np.mean(eroded[labels == 0]) < np.mean(eroded[labels == 1]):
                nucleus_label = 0
        else:
                nucleus_label = 1
        nucleus_mask = (labels == nucleus_label).astype(np.uint8)*255

        # CONTOUR
        blur = cv2.GaussianBlur(nucleus_mask, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = crop.copy()
        nucleus_area_px = 0
        if contours:
                cnt = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                nucleus_area_px = np.pi * radius ** 2
                nucleus_area_um2 = nucleus_area_px * area_convert
                print(f"Area: {nucleus_area_um2:.2f} µm²")

        # Ve
        cv2.circle(vis, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        cv2.circle(vis, (int(cx)), int(cy), 2, (0, 0, 255), 3)

        # Plot
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        ax[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original")

        ax[1].imshow(eroded, cmap='gray')
        ax[1].set_title("Erosion")

        ax[2].imshow(labels, cmap='viridis')
        ax[2].set_title("KMeans (2 lớp)")

        ax[3].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax[3].set_title("Nucleus + Circle")

        for a in ax:
                a.axis('off')

        plt.tight_layout()
        plt.show()