import torch
from torch import nn
import math
import torch
from torch.autograd import Variable

class Embedder(nn.Module):
    """ 
    Simple class to handle learned embeddings to convert the input/output tokens to vectors
    of dimension d_model 

    Arguments:
        d_model: hidden size of the Transformer
        vocab_size: input/output vocab_size
    
    Returns:
        A Tensor of shape [batch_size, seq_len, d_model]
    """ 
    def __init__(self, d_model, vocab_size):
        super(Embedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embedding(x) 

    
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
    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):

        # described in the paper (make the embeddings a little larger)
        x = x * math.sqrt(self.d_model) 

        #obtain the positional encodings (these encodings are fixed and not learnt during training)
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False) 

        if x.is_cuda:
            pe.cuda()

        # add the word embeddings with positional encodings
        x = x + pe

        return self.dropout(x)