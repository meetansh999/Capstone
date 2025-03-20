import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from model_Deeplab import get_model  
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ✅ Send to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load Model
model = get_model(num_classes=5, pretrained=True).to(device)

# ✅ Dataset Class
class OilSpillDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.augment = augment

        self.color_map = {
            (0, 0, 0): 0,       # Sea Surface (Black)
            (0, 255, 255): 1,   # Oil Spill (Cyan)
            (255, 0, 0): 2,     # Look-Alike (Red)
            (153, 76, 0): 3,    # Ship (Brown)
            (0, 153, 0): 4      # Land (Green)
        }

        self.img_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=60),
            transforms.ElasticTransform(alpha=75.0, sigma=2.0, interpolation=transforms.InterpolationMode.NEAREST, fill=0),
            transforms.RandomResizedCrop(size=224, scale=(0.75, 1.0)),  # Random crop then resize back to 224x224


        ])

    def _convert_mask(self, mask):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask_indices = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        for color, class_idx in self.color_map.items():
            color_array = np.array(color, dtype=np.uint8)
            mask_indices[np.all(mask == color_array, axis=-1)] = class_idx

        return mask_indices

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # ✅ Read Grayscale Image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Keep grayscale channel
        image = torch.tensor(image, dtype=torch.float32)

        # ✅ Read & Convert Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask = self._convert_mask(mask)
        mask = torch.tensor(mask, dtype=torch.long)

        if self.augment:
            stacked = torch.cat([image, mask.unsqueeze(0)], dim=0)
            stacked = self.img_transforms(stacked)
            image, mask = stacked[0:1], stacked[1]

        return image, mask

    def __len__(self):
        return len(self.image_filenames)


# ✅ Define Paths
train_image_dir = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\train\images"
train_mask_dir = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\train\masks"
val_image_dir = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\val\images"
val_mask_dir = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\val\masks"

# ✅ Loss Functions
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 2.5, 1.6, 2.0, 1.0], device=device))

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        smooth = 1.0
        preds = torch.softmax(preds, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=5).permute(0, 3, 1, 2).float()

        intersection = torch.sum(preds * targets, dim=(2, 3))
        union = torch.sum(preds, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

dice_loss = DiceLoss()

if __name__ == "__main__":
    train_dataset = OilSpillDataset(train_image_dir, train_mask_dir, augment=True)
    val_dataset = OilSpillDataset(val_image_dir, val_mask_dir, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    num_epochs = 40
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler("cuda")

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # ✅ Forward Pass
            with torch.amp.autocast("cuda"):
                outputs = model(images) # ✅ Keep ['out'] since your model returns a dictionary
                masks = masks.clamp(0, 4).long()
                loss1 = loss_fn(outputs, masks)
                loss2 = dice_loss(outputs, masks)
                loss = loss1 + loss2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)  # ✅ Keep ['out'] since your model returns a dictionary
                masks = masks.clamp(0, 4).long()
                loss1 = loss_fn(outputs, masks)
                loss2 = dice_loss(outputs, masks)
                loss = loss1 + loss2
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    torch.save(model.state_dict(), "train_DeeplabV3.pth")
    print("Training complete! Model saved.")
