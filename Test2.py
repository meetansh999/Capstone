
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
# import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from model2 import UNetMobileNetV3  

# Set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Paths
test_image_dir = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\test\images"
test_mask_dir = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\test\masks"
model_path = "best_modelnow.pth"
output_dir = "best_setmaybeoo"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Define Color to Class Mapping
COLOR_MAP = {
    (0, 0, 0): 0,       # Sea Surface (Black)
    (0, 255, 255): 1,   # Oil Spill (Cyan)
    (255, 0, 0): 2,     # Look-Alike (Red)
    (153, 76, 0): 3,    # Ship (Brown)
    (0, 153, 0): 4      # Land (Green)
}

# Reverse mapping for visualization
CLASS_TO_COLOR = {v: k for k, v in COLOR_MAP.items()}

# Load Model
model = UNetMobileNetV3(num_classes=5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set model to evaluation mode

# Define Dataset Class
class OilSpillTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def _convert_mask(self, mask):
        """Convert an RGB mask to class indices."""
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask_indices = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for color, class_idx in COLOR_MAP.items():
            color_array = np.array(color, dtype=np.uint8)
            mask_indices[np.all(mask == color_array, axis=-1)] = class_idx
        return mask_indices

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load grayscale image and normalize
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Keep (1, H, W)
        image = torch.tensor(image, dtype=torch.float32)

        # Load and process ground truth mask
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask = self._convert_mask(mask)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

    def __len__(self):
        return len(self.image_filenames)

# Load Test Data
test_dataset = OilSpillTestDataset(test_image_dir, test_mask_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Compute Metrics
def compute_mIoU(preds, targets, num_classes=5):
    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        true_mask = targets == cls
        intersection = (pred_mask & true_mask).sum()
        union = (pred_mask | true_mask).sum()
        if union == 0:
            ious.append(1.0)  # Avoid division by zero
        else:
            ious.append(intersection / union)
    return np.mean(ious)

def compute_pixel_accuracy(preds, targets):
    return (preds == targets).sum() / targets.size  # Use .size for NumPy arrays

def compute_dice_coefficient(preds, targets):
    smooth = 1e-6
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return (2. * intersection + smooth) / (union + smooth)

def convert_to_color(mask):
    """Convert class index mask to RGB color mask."""
    h, w = mask.shape
    print(mask.shape)
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in CLASS_TO_COLOR.items():
        color_mask[mask == class_idx] = color
    return color_mask

# Evaluate Model & Visualize Predictions
total_miou, total_accuracy, total_dice, num_samples = 0, 0, 0, 0

with torch.no_grad():
    for idx, (images, masks) in enumerate(test_loader):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)  # Get class with highest probability

        miou = compute_mIoU(preds.cpu().numpy(), masks.cpu().numpy())
        accuracy = compute_pixel_accuracy(preds.cpu().numpy(), masks.cpu().numpy())
        dice = compute_dice_coefficient(preds.cpu().numpy(), masks.cpu().numpy())

        total_miou += miou
        total_accuracy += accuracy
        total_dice += dice
        num_samples += 1

        print(f"Sample {num_samples}: mIoU = {miou:.4f}, Accuracy = {accuracy:.4f}, Dice Coefficient = {dice:.4f}")

        # Visualization for first 5 samples
        if idx < 60:
            image_np = images.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
            image_np = image_np.astype(np.uint8)
            gt_mask_np = masks.squeeze(0).cpu().numpy()
            pred_mask_np = preds.squeeze(0).cpu().numpy()

            gt_mask_color = convert_to_color(gt_mask_np)
            pred_mask_color = convert_to_color(pred_mask_np)

            # Plot and save visualization
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(image_np, cmap="gray")
            axs[0].set_title("Input Image")
            axs[0].axis("off")

            axs[1].imshow(gt_mask_color)
            axs[1].set_title("Ground Truth Mask")
            axs[1].axis("off")

            axs[2].imshow(pred_mask_color)
            axs[2].set_title("Predicted Mask")
            axs[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{idx+1}.png"))
            plt.close()

# Print Final Results
avg_miou = total_miou / num_samples
avg_accuracy = total_accuracy / num_samples
avg_dice = total_dice / num_samples

print("\nModel Evaluation Results:")
print(f"Mean IoU: {avg_miou:.4f}")
print(f"Pixel Accuracy: {avg_accuracy:.4f}")
print(f"Dice Coefficient: {avg_dice:.4f}")

print(f"Visualization saved in '{output_dir}'")
print("Testing complete!")
