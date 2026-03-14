"""XCNN model with residual blocks for terrain recognition"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for XCNN.
    Improves gradient flow and enables deeper networks.
    """
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        """Forward pass with skip connection"""
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection (downsample if needed)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out


class TerrainXCNN(nn.Module):
    """
    eXtended CNN (XCNN) with Residual Blocks for terrain recognition.
    Based on ResNet architecture with skip connections.
    Classifies terrain into: Desert, Forest, Mountain, Plains
    
    Features:
    - Skip connections for better gradient flow
    - Deeper network for more complex pattern learning
    - Multi-scale feature extraction
    - Batch normalization for stable training
    """
    
    def __init__(self, num_classes=4, num_blocks=[2, 2, 2, 2]):
        super(TerrainXCNN, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with dropout for classification
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, out_channels, blocks, stride=1):
        """Create a layer with multiple residual blocks"""
        downsample = None
        
        # Create downsampling layer if stride > 1 or channel mismatch
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        # First block with potential downsampling
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        # Remaining blocks without downsampling
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through XCNN"""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers with skip connections
        x = self.layer1(x)   # 64 channels
        x = self.layer2(x)   # 128 channels
        x = self.layer3(x)   # 256 channels
        x = self.layer4(x)   # 512 channels
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


# Keep backward compatibility
TerrainCNN = TerrainXCNN


def create_model(num_classes=4, device='cpu'):
    """
    Create and return the model on specified device
    
    Args:
        num_classes: Number of output classes
        device: Device to load model ('cpu' or 'cuda')
    
    Returns:
        Model loaded on specified device
    """
    model = TerrainCNN(num_classes=num_classes)
    model = model.to(device)
    return model
