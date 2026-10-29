---

### 2. `02-cnn-dog-cat/README.md`
```markdown
# CNN for Dog–Cat Image Classification

A PyTorch implementation of a Convolutional Neural Network featuring three convolutional blocks for processing and classifying image data.

## Architecture
- **Block 1:** Conv2d ($3 \rightarrow 32$ channels), BatchNorm, ReLU, MaxPool2d
- **Block 2:** Conv2d ($32 \rightarrow 64$ channels), BatchNorm, ReLU, MaxPool2d
- **Block 3:** Conv2d ($64 \rightarrow 128$ channels), BatchNorm, ReLU, MaxPool2d
- **Classifier:** Flatten $\rightarrow$ Linear ($256$ units, ReLU) $\rightarrow$ Dropout ($0.5$) $\rightarrow$ Linear ($1$ unit, Sigmoid)

## Usage
Run the script to initialize the model and verify dummy tensor batch processing:
```bash
python train.py
