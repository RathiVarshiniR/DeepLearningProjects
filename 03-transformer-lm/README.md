### 3. `03-transformer-lm/README.md`
```markdown
# Transformer-Based Next-Token Prediction Model

A lightweight causal language model built using PyTorch's `nn.TransformerEncoder` and basic self-attention mechanisms.

## Key Components
- **Token Embedding & Positional Embedding:** Encodes sequence tokens with learnable positions.
- **Causal Masking:** Ensures the model only attends to preceding tokens during self-attention, preventing future token lookahead.
- **Transformer Encoder Layers:** Stacked multi-head self-attention blocks.
- **Vocabulary Projection:** Linear layer mapping hidden dimensions back to vocabulary logits.

## Usage
Run the script to test the model with a dummy input sequence:
```bash
python train.py
