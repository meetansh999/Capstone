# ResNet34 + U-Net 

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34_UNet(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.3):
        super(ResNet34_UNet, self).__init__()

        # Load pretrained ResNet34 as encoder
        resnet = models.resnet34(weights="IMAGENET1K_V1")  # Load pretrained weights

        # ✅ Modify First Convolution Layer for Grayscale Images
        self.encoder_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # (1-channel input)

        # Encoder Layers (Extract Features from ResNet34)
        self.encoder = nn.Sequential(
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # Decoder (U-Net Upsampling)
        self.decoder = nn.Sequential(
            self._upsample_block(512, 256, dropout_p),
            self._upsample_block(256, 128, dropout_p),
            self._upsample_block(128, 64, dropout_p),
            self._upsample_block(64, 32, dropout_p),
            self._upsample_block(32, 16, dropout_p)
        )

        # Final Segmentation Layer
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels, dropout_p):
        """Helper function for upsampling in U-Net decoder."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),  # ✅ Dropout added for regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

    def forward(self, x):
        """Forward Pass"""
        x = self.encoder_conv1(x)  # Process grayscale images
        features = self.encoder(x)  # Extract features
        out = self.decoder(features)  # Decode features into segmentation map
        out = self.segmentation_head(out)  # Final segmentation prediction
        return out

# ✅ Model Test
if __name__ == "__main__":
    model = ResNet34_UNet(num_classes=5, dropout_p=0.3).to("cuda" if torch.cuda.is_available() else "cpu")

    from torchsummary import summary
    summary(model, (1, 224, 224))  # ✅ Grayscale Input
