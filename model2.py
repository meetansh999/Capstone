
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetMobileNetV3(nn.Module):
    def __init__(self, num_classes, dropout_p=0.4):  # Introduce dropoout probability
        super(UNetMobileNetV3, self).__init__()
        
        self.model = smp.Unet(
            encoder_name="timm-mobilenetv3_large_100",
            encoder_weights="imagenet",
            in_channels=1,  # Ensure grayscale input
            classes=num_classes
        )

        # Modify the first convolution layer to accept grayscale 
        self.model.encoder.conv_stem = nn.Conv2d(
            in_channels=1,  
            out_channels=16,  
            kernel_size=3, 
            stride=2, 
            padding=1, 
            bias=False
        )

        
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        """Manually apply dropout to encoder activations"""
        enc_features = self.model.encoder(x)  # Get encoder features
        
        # Apply dropout at multiple levels of encoder
        enc_features = [self.dropout(f) for f in enc_features]

        out = self.model.decoder(*enc_features)
        return self.model.segmentation_head(out)

# Function to initialize and return the model
def get_model(num_classes, dropout_p=0.4):
    return UNetMobileNetV3(num_classes=num_classes, dropout_p=dropout_p)

# main implementation
if __name__ == "__main__":
    num_classes = 5
    model = get_model(num_classes, dropout_p=0.4)  
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Print model summary
    from torchsummary import summary
    summary(model, (1, 256, 256))  
