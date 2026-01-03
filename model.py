import torch
import torch.nn as nn
from torchvision import models

def create_resnet_model(num_classes=6, pretrained=True):
    """
    Creates a ResNet50 model with custom final layer
    """
    model = models.resnet50(pretrained=pretrained)
    
    # Freeze early layers (optional - comment out for full training)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def create_mobilenet_model(num_classes=6, pretrained=True):
    """
    Creates a MobileNetV2 model (lighter alternative)
    """
    model = models.mobilenet_v2(pretrained=pretrained)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model


class CustomCNN(nn.Module):
    """
    Custom CNN for waste classification (lightweight option)
    """
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x