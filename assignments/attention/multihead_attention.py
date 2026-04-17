"""
Masked Multi-Head Attention Assignment (Student Version)
========================================================

Implement masked multi-head attention in pure PyTorch.

Reference: https://arxiv.org/pdf/1706.03762

Goal:
- project query/key/value
- split into heads
- compute scaled dot-product attention
- apply an attention mask (causal or padding)
- merge heads and project output

Complete all TODOs.
"""

import math

import torch
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """
    Expected shapes:
    - query: (batch, query_len, embed_dim)
    - key:   (batch, key_len, embed_dim)
    - value: (batch, key_len, embed_dim)
    - attn_mask: usually provided for causal/padding constraints
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        # TODO: validate that embed_dim is divisible by num_heads

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # TODO: create linear projections for q, k, v and output
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None

        # TODO: create dropout layer for attention weights
        self.attn_dropout = None

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Convert shape:
        (batch, seq_len, embed_dim) -> (batch, heads, seq_len, head_dim)
        """
        # TODO: reshape and transpose
        return x

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        Convert shape:
        (batch, heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        """
        # TODO: transpose back, then reshape
        return x

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        # TODO: project query/key/value and split heads

        # TODO: compute scaled dot-product attention scores
        # expected shape: (batch, heads, query_len, key_len)

        # TODO: 
        # Expand attn_mask to match scores shape: (batch, heads, query_len, key_len).
        # This makes the broadcasting that happens in masked_fill below explicit.



        # TODO: multiply attention weights by V to get context
        
        # TODO: apply dropout to attention weights.

        # TODO: merge heads and apply output projection
        output = None
        
        return output


def main() -> None:
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_heads = 4

    # TODO: instantiate multi-head attention.
    mha = None
    x = torch.randn(batch_size, seq_len, embed_dim)

    # TODO: create causal mask for autoregressive self-attention (torch.bool).
    causal_mask = None

    # TODO: run forward pass and print output shape
    # Expected shape: (batch_size, seq_len, embed_dim)

    print("Student assignment executed.")
    print("Complete all TODO sections to produce working multi-head attention.")


if __name__ == "__main__":
    main()
