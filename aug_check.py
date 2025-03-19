import torch
import cv2
import os
import numpy as np
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt

# ✅ Set Paths
IMAGE_DIR = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\train\images"
MASK_DIR = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\train\masks"

# ✅ Get Sample Image & Mask
image_filenames = sorted(os.listdir(IMAGE_DIR))
mask_filenames = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".png")])

sample_img_path = os.path.join(IMAGE_DIR, image_filenames[0])
sample_mask_path = os.path.join(MASK_DIR, mask_filenames[0])

# ✅ Load Grayscale Image & Convert to Tensor
image = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (224, 224)) / 255.0  # Normalize & Resize
image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

# ✅ Load & Convert Mask (Ensure RGB)
mask = cv2.imread(sample_mask_path, cv2.IMREAD_COLOR)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Convert to RGB
mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

# ✅ Augmentations (Same as Training)
augmentations = [
    ("Horizontal Flip", transforms.RandomHorizontalFlip(p=1.0)),
    ("Vertical Flip", transforms.RandomVerticalFlip(p=1.0)),
    ("Rotation", transforms.RandomRotation(degrees=30)),
    ("Gaussian Blur", transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))),
    ("Gaussian Noise", transforms.GaussianNoise(mean=0.0, sigma=0.05)),
    ("Elastic Transform", transforms.ElasticTransform(alpha=125, sigma=2, interpolation=transforms.InterpolationMode.BILINEAR, fill= 0))
]

# ✅ Prepare Plot
fig, axes = plt.subplots(len(augmentations) + 1, 2, figsize=(8, 12))
axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("Original Image")
axes[0, 1].imshow(mask)
axes[0, 1].set_title("Original Mask")

# ✅ Apply Each Augmentation & Display
for i, (name, transform) in enumerate(augmentations):
    aug_img = transform(image_tensor).squeeze(0).numpy()
    
    # ✅ Ensure Mask is Tensor Before Augmenting
    mask_tensor = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W)
    aug_mask = transform(mask_tensor).permute(1, 2, 0).numpy().astype(np.uint8)  # Convert back
    
    # ✅ Plot Augmented Image & Mask
    axes[i + 1, 0].imshow(aug_img, cmap="gray")
    axes[i + 1, 1].imshow(aug_mask)
    
    axes[i + 1, 0].set_title(name)
    axes[i + 1, 1].set_title(name)

# ✅ Remove Axis Labels
for ax in axes.flatten():
    ax.axis("off")

plt.tight_layout()
plt.show()
