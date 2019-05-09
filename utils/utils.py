""" A set of utitilies of NMT Project """
import torch
import json
import os
import shutil
import logging
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

    Returns:
        A torch ModuleList that contains a `module` repeated N times
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def mask_invalid_positions(max_seq_len):
    """ 
    Mask out invalid positions for Masked Multi-head Attention 
    Arguments: 
        max_seq_len: Maximum length of sequence in a batch 

    Returns:
        An attention mask of size [seq_len, seq_len]
    """
    attn_shape = (1, max_seq_len, max_seq_len)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def make_tgt_mask(tgt, tgt_pad):
    """ 
    Make the mask for the target to hide padding and future words 

    Arguments:
        tgt: target sequence Tensor of shape [batch_size, seq_len]
        tgt_pad: id of the padding token

    Returns:
        A mask of size [batch_size, seq_len, seq_len] 
    """
    tgt_mask = (tgt != tgt_pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        mask_invalid_positions(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def set_logger(log_path):
    """
    Set logger to log info in the terminal and file `log path`

    Arguments:
        log_path: where to log
    """

    logger = logging.getLogger()
    # INFO: confirmation that things are working as expected
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class RunningAverage():
    """ A class that maintains the running average of a quanity """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class HyperParams():
    """ Class that loads hyperparams for a particular `model` from a JSON file  """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """ Loads parameters from a JSON file """
        with open(json_path) as f:
            params = json.loads(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """ Give dict-like access to Params """
        return self.__dict__
