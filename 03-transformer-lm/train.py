import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformerLM(nn.Module):

  def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, embed_dim)
    self.pos_embed = nn.Embedding(max_len, embed_dim)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim, nhead=num_heads, batch_first=True
    )
    self.transformer = nn.TransformerEncoder(
        encoder_layer, num_layers=num_layers
    )
    self.fc_out = nn.Linear(embed_dim, vocab_size)

  def forward(self, idx):
    seq_len = idx.size(1)
    pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)

    # Token embeddings + Positional embeddings
    x = self.embed(idx) + self.pos_embed(pos)

    # Generate causal mask to prevent attending to future tokens
    mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
        idx.device
    )

    x = self.transformer(x, mask=mask, is_causal=True)
    logits = self.fc_out(x)
    return logits


# Configuration and Test
vocab_size = 100
embed_dim = 64
num_heads = 4
num_layers = 2
max_len = 50

model = SimpleTransformerLM(vocab_size, embed_dim, num_heads, num_layers, max_len)

# Dummy input sequence of batch_size=2, seq_len=10
input_seq = torch.randint(0, vocab_size, (2, 10))
logits = model(input_seq)
print("Logits shape (Batch, Seq_Len, Vocab_Size):", logits.shape)