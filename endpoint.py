# File to call on the model to create prediction for an input image
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from model2 import UNetMobileNetV3

# Default Directories -- TODO: DETERMINE WITH COMMAND LINE INPUT
image_path = "./testInput.jpg"
model_path = "./model2.pth"
doVisual = False

# TODO: Add handling for CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imagepath', help="the full or relative path to the image input")
parser.add_argument('-m', '--modelpath', help="the full or relative path to the image input")
parser.add_argument('-p', '--matplot', help='provides output plot of input and prediction for local testing', action='store_true')

args = parser.parse_args()

if(args.imagepath != None):
    print("Using input image at: ", args.imagepath)
    image_path = args.imagepath
if(args.modelpath != None):
    print("Using input model at: ", args.modelpath)
    model_path = args.modelpath
if(args.matplot):
    print("Will create output visual")
    doVisual = args.matplot

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Define Color to Class Mapping
COLOR_MAP = {
    #BGR
    (0, 0, 0): 0,       # Sea Surface (Black)
    (0, 255, 255): 1,   # Oil Spill (Cyan)
    (255, 0, 0): 2,     # Look-Alike (Red)
    (153, 76, 0): 3,    # Ship (Brown)
    (0, 153, 0): 4      # Land (Green)
}

# Reverse mapping for visualization
CLASS_TO_COLOR = {v: k for k, v in COLOR_MAP.items()}

def convert_to_color(mask):
    """Convert class index mask to RGB color mask."""
    print(mask.shape)
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in CLASS_TO_COLOR.items():
        color_mask[mask == class_idx] = color
    return color_mask

# Load Model
model = UNetMobileNetV3(num_classes=5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set model to evaluation mode

#TODO: Handle GeoTIFF to regular image filetype conversion here
#TODO: Handle NDArray Inputs? JSON inputs? How do we do ee.predictImage stuff?


with torch.no_grad():

    #REPLACE WITH JUST TAKING IN A GRAYSCALE??
    # Normalization transforms
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]

    # Load grayscale image and normalize
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Keep (1, H, W)
    image = torch.tensor(image, dtype=torch.float32)
    
    image = image.to(device)
    image = image.unsqueeze(0)
    outputs = model(image)
    preds = torch.argmax(outputs, dim=1)
    # TODO: Fix BGR/RGB conversion
    
    if(doVisual):
        image = image.squeeze().cpu()
        preds = preds.squeeze().cpu().numpy()
        print("Squeezed: ", preds.shape)

        pred_mask_color = convert_to_color(preds)
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image, cmap="gray")
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(pred_mask_color)
        axs[1].set_title("Prediction")
        axs[1].axis("off")


        plt.tight_layout()
        plt.show()

        plt.close()





    