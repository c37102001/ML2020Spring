import torch
import torch.nn as nn
import torch.nn.functional as F



class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(          # (b, 1, 32, 32)
            nn.Conv2d(1, 64, 3, 1, 1),      # (b, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                # (b, 64, 16, 16)

            nn.Conv2d(64, 128, 3, 1, 1),    # (b, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                # (b, 128, 8, 8)

            nn.Conv2d(128, 256, 3, 1, 1),   # (b, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                # (b, 256, 4, 4)

            nn.Conv2d(256, 256, 3, 1, 1),   # (b, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                # (b, 256, 2, 2)

            nn.Conv2d(256, 512, 3, 1, 1),   # (b, 256, 2, 2)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)                 # (b, 512, 1, 1)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()          # (b, 512)
        return x


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y