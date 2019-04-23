import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """
    Implementation of Scaled Dot Product Attention
    Calculate attention with the following equation:

    Attention(Q, K, V) = ((QK^T) / sqrt(d_k)) * V

    Arguments:
        attn_dropout: Amount of dropout to apply to the attention scores
    
    Returns: 
        A Tensor of shape [batch_size, num_heads, seq_len, d_model/num_heads]
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
       

class DotProductAttention(nn.Module):
    """ Implementation of Dot Product Attention """

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super().__init__()
        self.attention_scores = None
        self.key_layer = lambda x: x

    def forward(self, query, projected_keys, mask, values):
        assert mask is not None, "Mask is required inorder to perform attention"

        # compute score (dot product) between decoder hidden state and all encoder hidden states
        scores = self.score(query, projected_keys)  # [batch_size, 1, seq_len]

        # mask out invalid positions (PAD tokens)
        # the mask is of shape (batch_size, 1, seq_len)
        # Give pad tokens a -infinity scores
        scores.data.masked_fill_(mask == 0, -float('inf'))  # (batch_size, 1, seq_len)
        attention_scores = F.softmax(scores, dim=2)  # (batch_size, 1, seq_len)
        self.attention_scores = attention_scores

        # create context vector (weight average of attention scores and encoder hidden states)
        context_vector = torch.bmm(attention_scores, values)  # (batch_size, 1, hidden_size)

        return context_vector, attention_scores

    def score(self, query, keys):
        """ Compute a score (dot product) between the query and the keys """
        return torch.bmm(query, keys.transpose(1, 2))


class BahdanauAttention(nn.Module):
    """ Implementation of Bahdanau (MLP) Attention """

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super().__init__()
        # assumption is that the Encoder's outputs at each timestep are summed, instead of concatenated
        key_size = hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        self.attention_scores = None

    def forward(self, query, projected_keys, mask, values):
        assert mask is not None, "Mask is required inroder to perform attention"

        # project the query (decoder hidden state)
        query = self.query_layer(query)  # (batch_size, 1, D) ==> (batch_size, 1, hidden_size)

        # compute the attention scores
        scores = self.score(query, projected_keys)  # [batch_size, 1, seq_len]

        # mask out invalid positions (PAD tokens)
        # the mask is of shape (batch_size, 1, seq_len)
        # Give pad tokens a -infinity scores
        scores.data.masked_fill_(mask == 0, -float('inf'))  # (batch_size, 1, seq_len)

        attention_scores = F.softmax(scores, dim=2)  # (batch_size, 1, seq_len)
        self.attention_scores = attention_scores

        # value => (batch_size, seq_len, hidden_size)
        # attention_scores => (batch_size, 1, seq_len)
        context_vector = torch.bmm(attention_scores, values)  # (batch_size, 1, hidden_size)

        # context shape: [batch_size, 1, hidden_size], attention_scores: [batch_size, 1, seq_len]
        return context_vector, attention_scores

    def score(self, query, keys):
        """ Compute score (Bahdaunau) between the query and keys """
        scores = self.energy_layer(torch.tanh(query + keys))
        scores = scores.squeeze(2).unsqueeze(1)
        return scores
