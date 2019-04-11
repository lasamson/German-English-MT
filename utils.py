from torchtext import data, datasets
from torchtext.data import Field, BucketIterator
import spacy
import torch
import json
import os
import shutil
import logging

def load_dataset(data_path, min_freq=2, batch_size=128):
    """
    Returns iterators for the training/dev dataset

    Arguments:
        data_path: path of the dataset
        min_freq: min freq. needed to include a token in the vocabulary
        batch_size: size of the batch for BucketIterator
    """

    spacy_de = spacy.load("de")
    spacy_en = spacy.load("en")

    def tokenize_de(text):
        """ Tokenize German text """
        return [tok.text for tok in spacy_de.tokenizer(text)]
    def tokenize_en(text):
        """ Tokenize English text """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize=tokenize_de, init_token="<BOS>", eos_token="<EOS>")
    TRG = Field(tokenize=tokenize_en, init_token="<BOS>", eos_token="<EOS>")

    train_data, dev_data = datasets.TranslationDataset.splits(exts=(".de", ".en"),
                                    fields=(SRC, TRG), path=data_path, test=None, validation="dev")

    SRC.build_vocab(train_data, min_freq=min_freq)
    TRG.build_vocab(train_data, min_freq=min_freq)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, dev_iterator = BucketIterator.splits((train_data, dev_data),
                                                         batch_size=batch_size,
                                                         device=device)
    return train_iterator, dev_iterator, SRC, TRG

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

def load_checkpoint(checkpoint, model, optmimizer=None):
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


class HyperParams():
    """ Class that loads hyperparams from a JSON file  """
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
