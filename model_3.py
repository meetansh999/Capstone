
# import torch
# import torch.nn as nn
# import segmentation_models_pytorch as smp

# class UNetEfficientNet(nn.Module):
#     def __init__(self, num_classes, dropout_p=0.4):  # Introduce dropoout probability
#         super(UNetEfficientNet, self).__init__()
        
#         self.model = smp.Unet(
#             encoder_name="timm-mobilenetv3_large_100",
#             encoder_weights="imagenet",
#             in_channels=1,  # Ensure grayscale input
#             classes=num_classes
#         )

#         # Modify the first convolution layer to accept grayscale 
#         self.model.encoder.conv_stem = nn.Conv2d(
#             in_channels=1,  
#             out_channels=48,  
#             kernel_size=3, 
#             stride=2, 
#             padding=1, 
#             bias=False
#         )

        
#         self.dropout = nn.Dropout2d(p=dropout_p)

#     def forward(self, x):
#         """Manually apply dropout to encoder activations"""
#         enc_features = self.model.encoder(x)  # Get encoder features
        
#         # Apply dropout at multiple levels of encoder
#         enc_features = [self.dropout(f) for f in enc_features]

#         out = self.model.decoder(*enc_features)
#         return self.model.segmentation_head(out)

# # Function to initialize and return the model
# def get_model(num_classes, dropout_p=0.4):
#     return UNetEfficientNet(num_classes=num_classes, dropout_p=dropout_p)

# # main implementation
# if __name__ == "__main__":
#     num_classes = 5
#     model = get_model(num_classes, dropout_p=0.3)  
#     model = model.to("cuda" if torch.cuda.is_available() else "cpu")

#     # Print model summary
#     from torchsummary import summary
#     summary(model, (1, 256, 256))  


import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetEfficientNet(nn.Module):
    def __init__(self, num_classes, dropout_p=0.4):  # ✅ Dropout probability
        super(UNetEfficientNet, self).__init__()
        
        self.model = smp.Unet(
            encoder_name="timm-efficientnet-b0",  # ✅ Switched to B0
            encoder_weights="imagenet",
            in_channels=1,  # ✅ Grayscale input
            classes=num_classes
        )

        # ✅ Modify first convolution layer for grayscale input
        self.model.encoder.conv_stem = nn.Conv2d(
            in_channels=1,  
            out_channels=32,  # ✅ EfficientNet-B0 first layer uses 32 filters
            kernel_size=3, 
            stride=2, 
            padding=1, 
            bias=False
        )

        # ✅ Dropout layers
        self.dropout_encoder = nn.Dropout2d(p=dropout_p)
        self.dropout_decoder = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        """Apply Dropout in Encoder & Decoder"""
        enc_features = self.model.encoder(x)  # ✅ Extract encoder features
        
        # ✅ Apply dropout to high-level encoder activations
        enc_features = [
            self.dropout_encoder(f) if i >= len(enc_features) // 2 else f
            for i, f in enumerate(enc_features)
        ] if self.training else enc_features

        # ✅ Pass through decoder
        out = self.model.decoder(*enc_features)
        
        # ✅ Apply dropout to decoder outputs
        out = self.dropout_decoder(out) if self.training else out

        return self.model.segmentation_head(out)

# ✅ Function to initialize and return model
def get_model(num_classes, dropout_p=0.4):
    return UNetEfficientNet(num_classes=num_classes, dropout_p=dropout_p)

# ✅ Model Test
if __name__ == "__main__":
    num_classes = 5
    model = get_model(num_classes, dropout_p=0.4)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Print Model Summary
    from torchsummary import summary
    summary(model, (1, 224, 224))
