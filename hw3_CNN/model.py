import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Conv2d output size = floor((W + 2pad - ks) / stride)+1
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input size [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      # (64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # (64, 64, 64)

            nn.Conv2d(64, 128, 3, 1, 1),    # (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # (128, 32, 32)

            nn.Conv2d(128, 256, 3, 1, 1),   # (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # (256, 16, 16)

            nn.Conv2d(256, 512, 3, 1, 1),    # (512, 16, 16)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # (512, 8, 8)

            nn.Conv2d(512, 512, 3, 1, 1),   # (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          # (512, 4, 4)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

        self.fconv = nn.Sequential(
            nn.Conv2d(512, 11, 4)           # (11, 1, 1)
        )

    def forward(self, x):               # (b, 3, 128, 128)
        out = self.cnn(x)               # (b, 512, 4, 4)
        
        # Linear
        out = out.view(x.shape[0], -1)  # (b, 512*4*4)
        out = self.fc(out)              # (b, 11)
        
        # # FCN
        # out = self.fconv(out)           # (b, 11, 1, 1)
        # out = out.view(x.shape[0], -1)    # (b, 11)
        
        return out