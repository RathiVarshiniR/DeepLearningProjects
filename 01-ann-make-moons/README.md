# ANN for Binary Classification (`make_moons`)

A PyTorch implementation of a Multi-Layer Perceptron (Artificial Neural Network) designed to solve non-linear binary classification using Scikit-learn's synthetic `make_moons` dataset.

## Architecture
- **Input Layer:** 2 features
- **Hidden Layer 1:** 16 neurons with ReLU activation
- **Hidden Layer 2:** 8 neurons with ReLU activation
- **Output Layer:** 1 neuron with Sigmoid activation (Binary output)

## Training Setup
- **Loss Function:** Binary Cross-Entropy Loss (`nn.BCELoss`)
- **Optimizer:** Adam ($lr = 0.01$)
- **Epochs:** 200

## Usage
Run the training script:
```bash
python train.py
