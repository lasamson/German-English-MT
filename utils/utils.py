import torch
import json
import os
import shutil
import logging

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
    logger.setLevel(logging.INFO) # INFO: confirmation that things are working as expected

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
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

def batch_reverse_tokenization(batch, eos_index, itos):
    """
    Convert the token IDs to words in a batch

    Arguments:
        batch: a tensor containg the decoded examples (with word ids representing the sequence)
        eos_index: the index of </s> token
        itos: idx2word dictionary mapping for the target language
    """
    sentences = []
    for example in batch:
        sentence = []
        for token_id in example:
            token_id = int(token_id.item())
            if token_id == eos_index:
                break
            sentence.append(itos[token_id])
        sentences.append(sentence)
    return sentences


def output_decoded_sentences_to_file(outputs, model_dir, filename):
    """
    Output a lsit of decoded sentences to a file

    Arguments:
        outputs: list of decoded sentences from the model (translations)
        model_dir: directory of the `model`
        filename: name of the file to output the decoded sentences
    """

    filepath = os.path.join(model_dir + "/outputs/", filename)

    if not os.path.exists(model_dir + "/outputs"):
        os.mkdir(model_dir + "/outputs")

    with open(filepath, "w") as f:
        for sentence in outputs:
            sentence = " ".join(sentence)
 