import torch
import torch.nn as nn
import torch.nn.functional as F

class ECG1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for ECG Arrhythmia Classification.
    Accepts 1D signals of shape (Batch, 1, 187) and classifies into 5 classes.
    Provides feature extraction for 1D-FID calculation and latent space analysis.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 5, feature_dim: int = 64):
        super().__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 187 -> 93
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 93 -> 46
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 46 -> 1
        )
        
        self.fc_embed = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts penultimate embedding representation (Batch, feature_dim).
        Used for 1D-FID metric and Latent Space visualizer.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        h = h.squeeze(-1)
        features = self.fc_embed(h)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
