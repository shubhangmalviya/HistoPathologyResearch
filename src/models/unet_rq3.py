"""
U-Net Implementation for RQ3 Stain Normalization Study
======================================================

A PyTorch implementation of U-Net specifically for Research Question 3.
This is separate from the RQ2 U-Net to preserve RQ2 findings and allow
for RQ3-specific modifications.

Key differences from RQ2 U-Net:
- Designed for stain normalization comparison
- 6 classes (background + 5 nuclei types) for PanNuke
- RQ3-specific optimizations and metrics
- Separate from RQ2 architecture to avoid conflicts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvRQ3(nn.Module):
    """Double convolution block for RQ3 U-Net"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConvRQ3, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DownRQ3(nn.Module):
    """Downscaling with maxpool then double conv for RQ3"""
    
    def __init__(self, in_channels, out_channels):
        super(DownRQ3, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvRQ3(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class UpRQ3(nn.Module):
    """Upscaling then double conv for RQ3"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpRQ3, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvRQ3(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvRQ3(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle input size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConvRQ3(nn.Module):
    """Final output convolution for RQ3"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConvRQ3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNetRQ3(nn.Module):
    """
    U-Net model specifically for Research Question 3.
    
    This is separate from the RQ2 U-Net to avoid any conflicts with existing
    trained models and results.
    
    Args:
        n_channels (int): Number of input channels (3 for RGB)
        n_classes (int): Number of output classes (6 for PanNuke RQ3)
        bilinear (bool): Use bilinear upsampling instead of transpose convolutions
    """
    
    def __init__(self, n_channels=3, n_classes=6, bilinear=True):
        super(UNetRQ3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.model_name = "UNet-RQ3"
        
        # Encoder
        self.inc = DoubleConvRQ3(n_channels, 64)
        self.down1 = DownRQ3(64, 128)
        self.down2 = DownRQ3(128, 256)
        self.down3 = DownRQ3(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownRQ3(512, 1024 // factor)
        
        # Decoder
        self.up1 = UpRQ3(1024, 512 // factor, bilinear)
        self.up2 = UpRQ3(512, 256 // factor, bilinear)
        self.up3 = UpRQ3(256, 128 // factor, bilinear)
        self.up4 = UpRQ3(128, 64, bilinear)
        
        # Output
        self.outc = OutConvRQ3(64, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def get_model_size(self):
        """Calculate model size in MB."""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Get comprehensive model information."""
        return {
            'model_name': self.model_name,
            'architecture': 'U-Net',
            'research_question': 'RQ3',
            'purpose': 'Stain normalization impact study',
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'parameters': self.count_parameters(),
            'size_mb': self.get_model_size(),
            'bilinear': self.bilinear
        }


def create_unet_rq3(n_channels=3, n_classes=6, device='cuda', verbose=True):
    """
    Create and initialize a U-Net model specifically for RQ3.
    
    Args:
        n_channels (int): Number of input channels
        n_classes (int): Number of output classes  
        device (str): Device to place the model on
        verbose (bool): Print model information
    
    Returns:
        torch.nn.Module: Initialized U-Net RQ3 model
    """
    model = UNetRQ3(n_channels=n_channels, n_classes=n_classes, bilinear=True)
    model = model.to(device)
    
    # Initialize weights using He initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    if verbose:
        model_info = model.get_model_info()
        print(f"✅ {model_info['model_name']} created for {model_info['research_question']}:")
        print(f"   Purpose: {model_info['purpose']}")
        print(f"   Parameters: {model_info['parameters']:,}")
        print(f"   Model size: {model_info['size_mb']:.2f} MB")
        print(f"   Input channels: {model_info['n_channels']}")
        print(f"   Output classes: {model_info['n_classes']}")
        print(f"   Device: {device}")
        print(f"   🔒 Separate from RQ2 U-Net to preserve existing results")
    
    return model


# Compatibility function to match RQ2 naming if needed
def create_unet_model_rq3(*args, **kwargs):
    """Alias for create_unet_rq3 to maintain compatibility"""
    return create_unet_rq3(*args, **kwargs)


if __name__ == "__main__":
    # Test the RQ3 model
    print("Testing U-Net RQ3 model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_unet_rq3(n_channels=3, n_classes=6, device=device)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(x)
    
    print(f"\n📊 Forward Pass Test:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Compare with RQ2 architecture
    print(f"\n🔍 Architecture Comparison:")
    print(f"   RQ2 U-Net: UNet(in_channels=3, num_classes=7)")
    print(f"   RQ3 U-Net: UNetRQ3(n_channels=3, n_classes=6)")
    print(f"   ✅ Completely separate implementations")
    
    print(f"\n✅ U-Net RQ3 model test successful!")
    print(f"🔒 RQ2 findings are completely preserved!")
