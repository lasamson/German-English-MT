import torch
from torch import optim
import argparse, os, sys, shutil, logging
from utils.data_loader import load_dataset
from utils.utils import set_logger, HyperParams
from models.transformer.models import make_transformer
from utils.label_smoothing import LabelSmoothingLoss
from models.transformer.optim import ScheduledOptimizer
from models.transformer_2.Models import get_model
from trainer import Trainer

def main(params):
    """
    The main function for training the Seq2Seq model with Dot-Product Attention
    Arguments:
        params: hyperparameters for the model
        model_dir: directory of the model
        restore_file: restore file for the model
    """

    logging.info("Loading the datasets...")
    train_iter, dev_iter, DE, EN = load_dataset(params.data_path, params.min_freq, params.train_batch_size, params.dev_batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    logging.info("[DE Vocab Size]: {}, [EN Vocab Size]: {}".format(de_size, en_size))
    logging.info("- done.")

    params.tgt_vocab_size = en_size
    params.src_vocab_size = de_size
    params.pad_token = EN.vocab.stoi["<pad>"]

    device = torch.device('cuda' if params.cuda else 'cpu')
    params.heads = 8
    params.n_layers = 6
    params.dropout = 0.1
    params.load_weights = None
    params.device = -1

    # make the transformer model with the given parameters
    model = make_transformer(params)
    # model = get_model(params, de_size, en_size)
    
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    optimizer = ScheduledOptimizer(optim.Adam(model.parameters(), lr=params.lr), params.d_model, params.n_warmup_steps)
    criterion = LabelSmoothingLoss(.01, params.tgt_vocab_size, params.pad_token)
    trainer = Trainer(model, optimizer, criterion, params.epochs, train_iter, dev_iter, params)
    trainer.train()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Baseline Transformer Hyperparameters")
    p.add_argument("-data_path", type=str, help="location of data")
    p.add_argument("-model_dir", default="./experiments/transformer_baseline", help="Directory containing transformer experiments")
    p.add_argument("-restore_file", default=None, help="Name of the file in the model directory containing weights \
                   to reload before training")
    args = p.parse_args()

    # create an experiments folder for training the seq2seq model
    if not os.path.exists("./experiments/transformer_baseline/"):
        os.mkdir("./experiments/transfomer_baseline/")

    # Set the logger
    set_logger(os.path.join(args.model_dir, "train.log"))

    # load the params json file (if it exists)
    json_params_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_params_path), "No json configuration file found at {}".format(json_params_path)
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
    if params.cuda: torch.cuda.manual_seed(2)
    
    main(params)
