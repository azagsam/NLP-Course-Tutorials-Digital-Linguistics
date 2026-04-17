import math

import torch
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """
    Pure PyTorch implementation of masked multi-head attention.

    Expected shapes:
    - query: (batch, query_len, embed_dim)
    - key:   (batch, key_len, embed_dim)
    - value: (batch, key_len, embed_dim)
    - attn_mask: usually provided for causal/padding constraints
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        # (batch, seq_len, embed_dim) -> (batch, heads, seq_len, head_dim)
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # (batch, heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor
    ) -> Tensor:
        # Project inputs into Q/K/V spaces, then split embed_dim across heads.
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        # Raw compatibility scores per head: Q @ K^T.
        # Scale by sqrt(head_dim) to keep softmax numerically stable.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Expand attn_mask to match scores shape: (batch, heads, query_len, key_len).
        # This makes explicit the broadcasting that happens in masked_fill below.
        while attn_mask.dim() < 4:
            attn_mask = attn_mask.unsqueeze(0)

        # Mask out invalid positions with -inf so they get zero attention after softmax.
        scores = scores.masked_fill(~attn_mask, float("-inf"))

        # Normalize over key positions so each query attends to a distribution.
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights.
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values, then merge heads back to embed_dim.
        context = torch.matmul(attn_weights, v)
        output = self.out_proj(self._merge_heads(context))

        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    # Define dimensions.
    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_heads = 4

    # Instantiate multi-head attention and create dummy input.
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Causal mask for autoregressive self-attention (use torch.bool dtype).
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    # Forward pass
    output = mha(
        query=x,
        key=x,
        value=x,
        attn_mask=causal_mask,
    )
    print("Output shape:", output.shape)  # Should be (batch_size, seq_len, embed_dim)
