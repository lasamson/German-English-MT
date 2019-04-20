import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from utils.data_loader import load_dataset
from models.seq2seq import Encoder, AttentionDecoder, Decoder, Seq2Seq
from models.attention import DotProductAttention, BahdanauAttention
from utils.utils import HyperParams, set_logger, RunningAverage
import os, sys, shutil
import logging
import time
import math
from tqdm import tqdm

class Trainer(object):
    """
    Class to handle the training of Seq2Seq based models
    """
    def __init__(self, model, optimizer, num_epochs, train_iter, dev_iter, params):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.params = params
        self.epoch = 0
        self.max_num_epochs = num_epochs
        self.best_val_loss = float("inf")
    
    def train_epoch(self):
        """
        Train `model` for one epoch
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss(ignore_index=self.params.pad_token)
        loss_avg = RunningAverage()
        pp = RunningAverage()
        with tqdm(total=len(self.train_iter)) as t:
            for idx, batch in enumerate(self.train_iter):
                src, src_lengths = batch.src
                trg, trg_lengths = batch.trg
                src_mask = (src != self.params.pad_token).unsqueeze(-2)

                if self.params.cuda:
                    src, trg = src.cuda(), trg.cuda()

                output = self.model(src, trg, src_lengths, trg_lengths, src_mask, tf_ratio=self.params.teacher_forcing_ratio)
                output = output[:, :-1, :].contiguous().view(-1, self.params.vocab_size)
                trg = trg[:, 1:].contiguous().view(-1)

                assert output.size(0) == trg.size(0)

                self.optimizer.zero_grad()
                loss = criterion(output, trg)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.grad_clip)
                self.optimizer.step()

                # update the average loss
                loss_avg.update(loss.item())
                pp.update(math.exp(loss_avg()))
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()), pp='{}'.format(pp()))
                t.update()
                torch.cuda.empty_cache()
        return loss_avg(), pp()

    def validate(self):
        """
        Evaluate the loss of the `model` on the dev set
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=self.params.pad_token)
        loss_avg = RunningAverage()
        pp = RunningAverage()
        with tqdm(total=len(self.dev_iter)) as t:
            with torch.no_grad():
                for idx, batch in enumerate(self.dev_iter):
                    src, src_lengths = batch.src
                    trg, trg_lengths = batch.trg
                    src_mask = (src != self.params.pad_token).unsqueeze(-2)

                    if self.params.cuda:
                        src, trg = src.cuda(), trg.cuda()

                    output = self.model(src, trg, src_lengths, trg_lengths, src_mask, tf_ratio=0.0)
                    output = output[:, :-1, :].contiguous().view(-1, self.params.vocab_size)
                    trg = trg[:, 1:].contiguous().view(-1)

                    assert output.size(0) == trg.size(0)

                    loss = criterion(output, trg)
                    loss_avg.update(loss.item())
                    pp.update(math.exp(loss_avg()))
                    t.set_postfix(loss='{:05.3f}'.format(loss_avg()), pp='{}'.format(pp()))
                    t.update()
        return loss_avg(), pp()

    def train(self):
        logging.info("Starting training for {} epoch(s)".format(self.max_num_epochs - self.epoch))
        for epoch in range(self.max_num_epochs):
            self.epoch = epoch
            logging.info("Epoch {}/{}".format(epoch+1, self.max_num_epochs))

            epoch_start_time = time.time()
            train_loss_avg, train_pp_avg = self.train_epoch()
            epoch_end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(epoch_start_time, epoch_end_time)
            logging.info(f'Epoch: {epoch+1:02} | Avg Train Loss: {train_loss_avg} | Avg Train Perplexity: {train_pp_avg} | Time: {epoch_mins}m {epoch_secs}s')

            val_loss_avg, val_pp_avg = self.validate() 
            logging.info(f'Avg Val Loss: {val_loss_avg} | Avg Val Perplexity: {val_pp_avg}')

            is_best = val_loss_avg < self.best_val_loss

              # save checkpoint
            self.save_checkpoint({
                "epoch": epoch+1,
                "state_dict": self.model.state_dict(),
                "optim_dict": self.optimizer.state_dict()},
                is_best=is_best,
                checkpoint=self.params.model_dir+"/checkpoints/")

            if is_best:
                logging.info("- Found new lowest loss!")
                self.best_val_loss = val_loss_avg

    def epoch_time(self, start_time, end_time):
        """ Calculate the time to train a `model` on a single epoch """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def save_checkpoint(self, state, is_best, checkpoint):
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

    def load_checkpoint(self, checkpoint):
        """
        Loads model parameters (state_dict) from file_path. If optimizer is provided
        loads state_dict of optimizer assuming it is present in checkpoint

        Arguments:
            checkpoint: filename which needs to be loaded
            optimizer: resume optimizer from checkpoint
        """
        if not os.path.exists(checkpoint):
            raise ("File doesn't exist {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint["state_dict"])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optim_dict"])
        return checkpoint

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

    params.vocab_size = en_size
    params.pad_token = EN.vocab.stoi["<pad>"]

    device = torch.device('cuda' if params.cuda else 'cpu')
    
    encoder = Encoder(src_vocab_size=de_size, embed_size=params.embed_size,
                    hidden_size=params.hidden_size, input_dropout_p=params.input_dropout_p_enc, 
                    num_layers=params.n_layers_enc, dropout_p=params.dropout_p)

    if "attention" in vars(params):
        logging.info("Running Seq2Seq w/ ({0}) Attention...".format(params.attention))
        decoder = AttentionDecoder(trg_vocab_size=en_size, embed_size=params.embed_size,
                        hidden_size=params.hidden_size, input_dropout_p=params.input_dropout_p_dec, 
                        dropout_p=params.dropout_p, attention=params.attention, 
                        num_layers=params.n_layers_dec)
    else:
        logging.info("Running regular Seq2Seq model...")
        decoder = Decoder(trg_vocab_size=en_size, embed_size=params.embed_size,
                        hidden_size=params.hidden_size, input_dropout_p=params.input_dropout_p_dec, 
                        dropout_p=params.dropout_p, num_layers=params.n_layers_dec)

    model = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    trainer = Trainer(model, optimizer, params.epochs, train_iter, dev_iter, params)

    if params.restore_file:
        restore_path = os.path.join(params.model_dir+"/checkpoints/", params.restore_file)
        logging.info("Restoring parameters from {}".format(restore_path))
        trainer.load_checkpoint(restore_path)

    trainer.train()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Seq2Seq w/ Attention Hyperparameters")
    p.add_argument("-data_path", type=str, help="location of data")
    p.add_argument("-model_dir", default="./experiments/seq2seq", help="Directory containing seq2seq experiments")
    p.add_argument("-restore_file", default=None, help="Name of the file in the model directory containing weights \
                   to reload before training")
    args = p.parse_args()

    # create an experiments folder for training the seq2seq model
    if not os.path.exists("./experiments/seq2seq/"):
        os.mkdir("./experiments/seq2seq/")

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
