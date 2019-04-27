import torch
from torch import nn
import math
import torch
from torch.autograd import Variable

class Embedder(nn.Module):
    """ 
    Simple class to convert the input/output tokens to vectors (embeddings)
    of dimension d_model 

    Arguments:
        d_model: hidden size of the Transformer
        vocab_size: input/output vocab_size
    
    Returns:
        A Tensor of shape [batch_size, seq_len, d_model]
    """ 
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # make the embeddings a little bigger (from the paper)
    
class PositionalEncoder(nn.Module):
    """
    Class to handle Position Encodings.
    Input to this layer is are the word embeddings (Embedder)
    These positional encodings are added to these word embeddings
    and dropout is applied to the sum.

    Arguments:
        d_model: hidden size of the Transformer
        max_seq_len: Maximum sequence length
        dropout: Dropout to be applied after the sums of the embeddings and the positional encodings
    
    Returns:
        A Tensor of shape [batch_size, seq_len, d_model]
    """
    def __init__(self, d_model, max_seq_len=1000, input_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(input_dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0., max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) 

    def forward(self, x):

        #obtain the positional encodings (these encodings are fixed and not learnt during training)
        pe = Variable(self.pe[:,:x.size(1)], requires_grad=False) 

        # add the word embeddings with positional encodings
        x = x + pe

        return self.dropout(x)