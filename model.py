import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()

        # First convolution layer:
        # input = 1 grayscale image channel
        # output = 32 feature maps
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # Second convolution layer:
        # input = 32 feature maps
        # output = 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Max pooling shrinks image size by half each time
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7

        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x