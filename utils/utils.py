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
    tgt_mask = tgt_mask & Variable(
        mask_invalid_positions(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def save_checkpoint(state, is_best, checkpoint):
    """
    Save a checkpoint of the model

    Arguments:
        state: dictionary containing information related to the state of the training process
        is_best: boolean value stating whether the current model got the best val loss
        checkpoint: folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, "last.pth.tar")
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided
    loads state_dict of optimizer assuming it is present in checkpoint

    Arguments:
        checkpoint: filename which needs to be loaded
        model: model for which the parametesr are loaded
        optimizer: resume optimizer from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])
    return checkpoint


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


def tile(x, count, dim):
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.
    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
