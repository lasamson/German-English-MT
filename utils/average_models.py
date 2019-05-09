""" Script to average the weights of different checkpoints of model training """
import os
import glob
import torch


def average_checkpoints(model_dir, n):
    """
    Average the weights of model checkpoints
    This will average the last `n` checkpoints
    saved in the /checkpoints folder in the model_dir
    folder. It will save model in a new file

    Arguments:
        model_dir: directory of model
        n: number of checkpoints to averag

    Returns:
        A dictionary containing the average of the weights for all checkpoints
    """

    averaged = {}
    scale = 1 / n

    checkpoint_files = get_last_n_checkpoints(model_dir, n)

    for model_file in checkpoint_files:
        checkpoint = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        state_dict = checkpoint["state_dict"]
        for n, p in state_dict.items():
            if n in averaged:
                averaged[n].add_(scale * p)
            else:
                averaged[n] = scale * p
    return averaged


def get_last_n_checkpoints(model_dir, n):
    """ 
    Get the last `n` checkpoints from the model directory 

    Arguments:
        model_dir: directory of the model (eg. ./experiments/transformer_large)   
        n: number of checkpoints to get from the model_dir/checkpoints directory

    Returns:
        A List of the checkpoint file paths
    """

    checkpoint_dir = model_dir + "checkpoints/"

    checkpoint_files = [x for x in glob.glob(
        checkpoint_dir + "*") if os.path.isfile(x)]
    checkpoint_files.sort(key=os.path.getmtime)

    return checkpoint_files[-n:]
