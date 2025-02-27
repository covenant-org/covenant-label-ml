from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.misc import variant_to_config_mapping
from sam2.utils.visualization import show_masks
from PIL import Image
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt

url = "https://github.com/SauravMaheshkar/SauravMaheshkar/blob/main/assets/text2img/llama_spiderman_coffee.png?raw=true"

image = Image.open(requests.get(url, stream=True).raw)
image = np.array(image.convert("RGB"))

model = build_sam2(
    variant_to_config_mapping["tiny"],
    "sam2_hiera_tiny.pt"
)
image_predictor = SAM2ImagePredictor(model)
image_predictor.set_image(image)
input_point = np.array([[300, 600]])
input_label = np.array([1])
masks, scores, logits = image_predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=None,
    multimask_output=False
)

w, h = image.shape[1], image.shape[0]
mask = masks[0].astype(np.uint8)
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
coords = []
for obj in contours:
    for point in obj:
        coords.append([point[0][0], point[0][1]])
coords = np.array(coords, dtype=np.float32)
coords = cv2.approxPolyDP(coords, 2, True)
curve = []
for obj in coords:
    for point in obj:
        print(point)
        curve.append([point[0], point[1]])
curve = np.array(curve, dtype=np.float32)
print(curve)
plt.figure(figsize=(20, 20))
#plt.axis("equal")
plt.fill(curve[:, 0], curve[:, 1])
plt.show()
