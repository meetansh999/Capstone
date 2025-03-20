import torch
import torch.nn as nn
import torchvision.models.segmentation as models

class DeepLabV3PlusMobileNetV3(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(DeepLabV3PlusMobileNetV3, self).__init__()
        
        # ✅ Load DeepLabV3+ with MobileNetV3 backbone
        self.model = models.deeplabv3_mobilenet_v3_large(
            weights=models.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )

        # ✅ Modify the classifier to match the number of output classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

        # ✅ Fix BatchNorm layers (preventing small batch issue)
        self._fix_batch_norm(self.model)

        # ✅ Modify the first convolution layer for grayscale input
        backbone = self.model.backbone
        first_conv_layer = backbone["0"][0]  # Access the first Conv2D layer inside the backbone

        new_conv_layer = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1 for grayscale input
            out_channels=first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=first_conv_layer.bias is not None
        )

        # ✅ Copy pretrained weights (sum over RGB channels and average)
        new_conv_layer.weight.data = first_conv_layer.weight.data.sum(dim=1, keepdim=True)

        # ✅ Replace the first convolution layer with the modified one
        backbone["0"][0] = new_conv_layer

    def forward(self, x):
        return self.model(x)['out']  # Extract segmentation mask

    def _fix_batch_norm(self, model):
        """Convert all BatchNorm layers to use track_running_stats=False."""
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False  # Fixes small batch normalization issue

# ✅ Function to initialize the model
def get_model(num_classes=5, pretrained=True):
    return DeepLabV3PlusMobileNetV3(num_classes=num_classes, pretrained=pretrained)

# ✅ Model Test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(num_classes=5, pretrained=True).to(device)

    # Test with a grayscale image (1-channel)
    sample_input = torch.randn(2, 1, 224, 224).to(device)  # Make sure batch size > 1
    output = model(sample_input)
    
    print("Output shape:", output.shape)  # Expected: [2, num_classes, 224, 224]
