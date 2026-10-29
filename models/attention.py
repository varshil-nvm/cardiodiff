import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        
    def forward(self, q, k, v, mask=None):
        # q, k, v: [Batch, Seq_Len, Dim]
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, d_model=16, n_heads=2):
        super(SimpleMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections & split into heads
        Q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1) # Broadcast mask for multiple heads
            
        # Apply attention over multiple heads
        x, attn_weights = self.attention(Q, K, V, mask=mask)
        
        # Concatenate heads and project output
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(x), attn_weights
