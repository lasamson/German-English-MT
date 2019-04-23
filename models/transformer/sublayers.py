""" Implementation of the different sublayers in the Transformer Model """

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class ScaledDotProductAttention(nn.Module):
    """
    Implementation of Scaled Dot Product Attention
    """

    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
    
    def forward(self, query, key, value, mask):
        d_k = query.size(-1) # get the size of the query

        # compute unnormalized scores
        # query: [batch_size, num_heads, seq_len, d_k]
        # keys: [batch_size, num_heads, seq_len, d_k]
        # [batch_size, num_heads, seq_len, d_k] * [batch_size, num_heads, d_k, seq_len] => [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 

        # apply mask to scores if given
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)

        # compute normalized attention scores
        attention_scores = F.softmax(scores, dim=-1) # [batch_size, num_heads, seq_len, seq_len]

        # apply dropout to the attention scores
        attention_scores = self.attn_dropout(attention_scores) 

        # [batch_size, num_heads, seq_len, seq_len]  * [batch_size, num_heads, seq_len, d_model/num_heads] => [batch_size, num_heads, seq_len, d_model/num_heads]
        return torch.matmul(attention_scores, value), attention_scores
       

class MultiHeadAttention(nn.Module):
    """
    Apply Multi-head Attention to the input

    Arguments:
        d_model: size of the model (eg. 512)
        num_heads: number of attention heads
        dropout: Dropout probability (Should be non-zero only during training)
    """

    def __init__(self, d_model, num_heads, attn_dropout=0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by the number of attention heads: {}".format(num_heads))

        self.num_heads = num_heads

        # scaled dot product attention
        self.attention = ScaledDotProductAttention(attn_dropout)
        
        # projection matrices
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # the same mask is applied to `num_heads` heads
            mask = mask.unsqueeze(1) # [batch_size, 1, seq_len]

        # Pass key, queries, values through linear layer
        # [batch_size, seq_len, d_model]
        queries = self.query_linear(query) 
        keys = self.key_linear(key)  
        values = self.value_linear(value)

        # split the heads
        # [batch_size, num_heads, seq_len, d_model/num_heads]
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # apply attention to each head
        # [batch_size, num_heads, seq_len, d_model/num_heads]
        contexts, _ = self.attention(queries, keys, values, mask=mask)

        # now we need to merge all the heads
        # [batch_size, num_heads, seq_len, d_model/num_heads] => [batch_size, seq_len, d_model]
        contexts = self._merge_heads(contexts) 

        # apply linear layer to `contexts` to get outputs
        # [batch_size, seq_len, d_model] => [batch_size, seq_len, d_model]
        outputs = self.output_linear(contexts)

        return outputs

    def _merge_heads(self, x):
        """
        Merge the heads input `x` into the last dimension

        Arguments:
            x: input Tensor with shape [batch_size, num_heads, seq_len, d_model/num_heads]

        Returns:
            A Tensor with shape [batch_size, seq_len, d_model]
        """
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.num_heads)

    def _split_heads(self, x):
        """
        Split input `x` to add extra `num_heads` dimension

        Arguments:
            x: input Tensor with shape [batch_size, seq_len, d_model]

        Returns:
            A Tensor of shape [batch_size, num_heads, seq_len, d_model/num_heads]
        """

        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)
    
class LayerNorm(nn.Module):
    """
    Apply LayerNorm to input

    Arguments:
        d_model: size of the hidden representation of the transformer (eg. 512)
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normalize = (x-mean) / (std + self.eps)
        return self.gamma * normalize + self.beta 

class PositionwiseFeedForwardNet(nn.Module):
    """
    Apply a linear + relu + linear on each timestep of the input
    
    The dimensionality of the input and output is d_model = 512 and the 
    inner-layer has dimensionality d_ff = 2048
    
    Arguments:
        d_model: size of the hidden representation of the Transformer (eg. 512)
        d_ff: hidden size representation of position wise feedforward net (eg. 2048)
    """
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear_1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        return out