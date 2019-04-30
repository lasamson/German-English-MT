import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from utils.data_loader import load_dataset
from models.seq2seq import make_seq2seq_model
from utils.label_smoothing import LabelSmoothingLoss
from models.transformer.optim import ScheduledOptimizer
from utils.utils import HyperParams, set_logger, RunningAverage
from utils.trainer import Trainer
import os
import sys
import shutil
import logging
import time
import math
from tqdm import tqdm


def main(params):
    """
    The main function for training the Seq2Seq model with Dot-Product Attention
    Arguments:
        params: hyperparameters for the model
        model_dir: directory of the model
        restore_file: restore file for the model
    """

    logging.info("Loading the datasets...")
    train_iter, dev_iter, DE, EN = load_dataset(
        params.data_path, params.train_batch_size, params.dev_batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    logging.info(
        "[DE Vocab Size]: {}, [EN Vocab Size]: {}".format(de_size, en_size))
    logging.info("- done.")

    params.src_vocab_size = de_size
    params.tgt_vocab_size = en_size

    params.pad_token = EN.vocab.stoi["<pad>"]

    device = torch.device('cuda' if params.cuda else 'cpu')
    params.device = device

    # make the Seq2Seq model
    model = make_seq2seq_model(params)

    # default optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    if params.model_type == "Transformer":
        criterion = LabelSmoothingLoss(
            params.label_smoothing, params.tgt_vocab_size, params.pad_token).to(device)
        optimizer = ScheduledOptimizer(optimizer=optimizer, d_model=params.hidden_size,
                                       factor=2, n_warmup_steps=params.n_warmup_steps)
        scheduler = None
    else:
        criterion = nn.NLLLoss(reduction="sum", ignore_index=params.pad_token)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.1, verbose=True)

    # intialize the Trainer
    trainer = Trainer(model, optimizer, scheduler, criterion,
                      params.epochs, train_iter, dev_iter, params)

    if params.restore_file:
        restore_path = os.path.join(
            params.model_dir+"/checkpoints/", params.restore_file)
        logging.info("Restoring parameters from {}".format(restore_path))
        Trainer.load_checkpoint(model, restore_path, optimizer)

    # train the model
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Seq2Seq w/ Attention Hyperparameters")
    p.add_argument("-data_path", type=str, help="location of data")
    p.add_argument("-model_dir", default="./experiments/seq2seq",
                   help="Directory containing seq2seq experiments")
    p.add_argument("-restore_file", default=None, help="Name of the file in the model directory containing weights \
                   to reload before training")
    args = p.parse_args()

    # Set the logger
    set_logger(os.path.join(args.model_dir, "train.log"))

    # load the params json file (if it exists)
    json_params_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(
        json_params_path), "No json configuration file found at {}".format(json_params_path)
    params = HyperParams(json_params_path)

    # add extra information to the params dictionary related to the training of the model
    params.data_path = args.data_path
    params.model_dir = args.model_dir
    params.restore_file = args.restore_file

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    logging.info("Using GPU: {}".format(params.cuda))

    # manual seed for reproducing experiments
    torch.manual_seed(2)
    if params.cuda:
        torch.cuda.manual_seed(2)
    main(params)
