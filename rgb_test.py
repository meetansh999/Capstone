import cv2
import numpy as np

# Load a sample mask with a ship region
mask = cv2.imread("img_0092.png", cv2.IMREAD_COLOR)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

# Get unique colors in the mask
unique_colors = np.unique(mask_rgb.reshape(-1, mask_rgb.shape[2]), axis=0)
print("Unique Colors in Mask:", unique_colors)
