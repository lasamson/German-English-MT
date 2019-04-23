import torch
from torch import nn
import copy
import numpy as np
from torch.autograd import Variable

def get_clones(module, N):
    """ 
    Produce N identical layers 
    Arguments:
        module: the module (layer) to repeat N times 
        N: number of identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def mask_invalid_positions(max_seq_len):
    """ 
    Mask out invalid positions for Masked Multi-head Attention 
    Arguments: 
        max_seq_len: Maximum length of sequence in a batch 

    Returns:
        an attention mask of size [seq_len, seq_len]
    """
    attn_shape = (1, max_seq_len, max_seq_len)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0

def make_tgt_mask(tgt, tgt_pad):
    """ 
    Make the mask for the target to hide padding and future words 

    Arguments:
        :tgt: target sequence Tensor batch_size, seq_len]
        tgt_pad: id of the padding token

    Returns:
        a mask of size [batch_size, seq_len, seq_len] 
    """
    tgt_mask = (tgt != tgt_pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(mask_invalid_positions(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
