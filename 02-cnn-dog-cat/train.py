import torch
import torch.nn as nn
import torch.optim as optim


class DogCatCNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.features = nn.Sequential(
        # Block 1: Conv Layer 1
        nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        ),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 64 x 64 (for 128x128)
        # Block 2: Conv Layer 2
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Output: 64 x 32 x 32
        # Block 3: Conv Layer 3
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Output: 128 x 16 x 16
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 16 * 16, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid(),  # Binary output: 0 for Cat, 1 for Dog
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x


# Instantiate model and verify structure
model = DogCatCNN()
print(model)

# Example dummy forward pass with batch of 2 images (3 channels, 128x128 pixels)
dummy_input = torch.randn(2, 3, 128, 128)
output = model(dummy_input)
print("Output shape:", output.shape)