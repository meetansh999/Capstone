import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
import os
import cv2
import numpy as np
from model2 import UNetMobileNetV3  
# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# send to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class OilSpillDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.augment = augment

        # RGB color to class index mapping
        self.color_map = {
            (0, 0, 0): 0,       # Sea Surface (Black)
            (0, 255, 255): 1,   # Oil Spill (Cyan)
            (255, 0, 0): 2,     # Look-Alike (Red)
            (153, 76, 0): 3,    # Ship (Brown)
            (0, 153, 0): 4      # Land (Green)
        }


        # ✅ Geometric Transforms (Apply to Both Images & Masks)
        self.img_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomVerticalFlip(p=0.4),
            transforms.RandomRotation(degrees=45),
            transforms.ElasticTransform(alpha=75, sigma=2, interpolation=transforms.InterpolationMode.NEAREST, fill=0),
            transforms.RandomCrop(size=(180, 180)),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
        ])

        # # ✅ Noise & Blur (Apply Only to Images)
        # self.img_transforms = transforms.Compose([
        #     transforms.GaussianNoise(mean=0.0, sigma=0.05),
        #     transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        # ])
    def _convert_mask(self, mask):
        """Convert an RGB mask to class indices based on predefined color mappings."""
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # color map conversion according to model
        mask_indices = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        for color, class_idx in self.color_map.items():
            color_array = np.array(color, dtype=np.uint8)  
            mask_indices[np.all(mask == color_array, axis=-1)] = class_idx

        return mask_indices

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load grayscale image and convert to tensor
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)   
        image = torch.tensor(image, dtype=torch.float32)  

        # Load and convert RGB mask
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask = self._convert_mask(mask)  # Convert mask to class indices
        mask = torch.tensor(mask, dtype=torch.long)

        
        if self.augment:
            stacked = torch.cat([image, mask.unsqueeze(0)], dim=0)  
            stacked = self.img_transforms(stacked)  
            image, mask = stacked[0:1], stacked[1]  

        return image, mask

    def __len__(self):
        return len(self.image_filenames)


# Define Paths
train_image_dir = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\train\images"
train_mask_dir = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\train\masks"
val_image_dir = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\val\images"
val_mask_dir = r"C:\Users\Kharb\Desktop\capstone\CSCI447_FinalProject-main\CSCI447_FinalProject-main\data\input\val\masks"

# Define Loss Functions
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 2.7, 1.9, 2.2, 1.1], device=device))  # class weights


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        smooth = 1.0
        preds = torch.softmax(preds, dim=1)  # Convert logits to probabilities
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

    
    model = UNetMobileNetV3(num_classes=5).to(device)  

    num_epochs = 35

    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-2)  

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler("cuda")

    # Store lossess as lists 
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # Using AMP for faster training
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                masks = masks.clamp(0, 4).long()
                loss1 = loss_fn(outputs, masks)
                loss2 = dice_loss(outputs, masks)
                loss = loss1 + loss2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        del images, masks, outputs, loss

        torch.cuda.empty_cache()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                masks = masks.clamp(0, 4).long()
                loss1 = loss_fn(outputs, masks)
                loss2 = dice_loss(outputs, masks)
                loss = loss1 + loss2
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if epoch % 5 == 0:
            torch.cuda.empty_cache()

    # save model
    torch.save(model.state_dict(), "train2_4.pth")
    print("Training complete! Model saved.")
