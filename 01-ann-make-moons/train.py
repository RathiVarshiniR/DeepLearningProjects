import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate and Prepare Data
X, y = sklearn.datasets.make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)


# 2. Define ANN Architecture
class MoonClassifierANN(nn.Module):

  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )

  def forward(self, x):
    return self.network(x)


model = MoonClassifierANN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Training Loop
epochs = 200
for epoch in range(epochs):
  model.train()
  optimizer.zero_grad()
  predictions = model(X_train_t)
  loss = criterion(predictions, y_train_t)
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 50 == 0:
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 4. Evaluation
model.eval()
with torch.no_grad():
  test_preds = model(X_test_t)
  test_preds_class = (test_preds >= 0.5).float()
  accuracy = (test_preds_class == y_test_t).float().mean()
  print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")