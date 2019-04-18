import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class DotProductAttention(nn.Module):
    """ Implementation of Dot Product Attention """
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super().__init__()
        self.attention_scores = None
        self.key_layer = lambda x: x
         
    def forward(self, query, projected_keys, mask, values):
        assert mask is not None, "Mask is required inorder to perform attention"

        # compute score (dot product) between decoder hidden state and all encoder hidden states
        scores = self.score(query, projected_keys) # [batch_size, 1, seq_len]

        # mask out invalid positions (PAD tokens)        
        # the mask is of shape (batch_size, 1, seq_len)
        # Give pad tokens a -infinity scores  
        scores.data.masked_fill_(mask == 0, -float('inf')) # (batch_size, 1, seq_len)
        attention_scores = F.softmax(scores, dim=2) # (batch_size, 1, seq_len)
        self.attention_scores = attention_scores

        # create context vector (weight average of attention scores and encoder hidden states)
        context_vector = torch.bmm(attention_scores, values) # (batch_size, 1, hidden_size)
        
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
        query = self.query_layer(query) # (batch_size, 1, D) ==> (batch_size, 1, hidden_size)

        # compute the attention scores
        scores = self.score(query, projected_keys) # [batch_size, 1, seq_len]

        # mask out invalid positions (PAD tokens)        
        # the mask is of shape (batch_size, 1, seq_len)
        # Give pad tokens a -infinity scores  
        scores.data.masked_fill_(mask == 0, -float('inf')) # (batch_size, 1, seq_len)

        attention_scores = F.softmax(scores, dim=2) # (batch_size, 1, seq_len)
        self.attention_scores = attention_scores

        # value => (batch_size, seq_len, hidden_size)
        # attention_scores => (batch_size, 1, seq_len)
        context_vector = torch.bmm(attention_scores, values) # (batch_size, 1, hidden_size)

        # context shape: [batch_size, 1, hidden_size], attention_scores: [batch_size, 1, seq_len]
        return context_vector, attention_scores 
    
    def score(self, query, keys):
        """ Compute score (Bahdaunau) between the query and keys """
        scores = self.energy_layer(torch.tanh(query + keys))
        scores = scores.squeeze(2).unsqueeze(1)
        return scores
