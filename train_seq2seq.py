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
from utils.utils import HyperParams, set_logger, load_checkpoint, save_checkpoint, RunningAverage, epoch_time
import os, sys
import logging
import time
import math

def evaluate_loss_on_dev(model, dev_iter, params):
    """
    Evaluate the loss of the `model` on the dev set
    Arguments:
        model: the neural network
        dev_iter: BucketIterator for the dev set
        params: hyperparameters for the `model`
    """

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=params.pad_token)
    loss_avg = RunningAverage()
    with torch.no_grad():
        for _, batch in enumerate(dev_iter):
            src, src_lengths = batch.src
            trg, trg_lengths = batch.trg
            src_mask = (src != params.pad_token).unsqueeze(-2)

            if params.cuda:
                src, trg = src.cuda(), trg.cuda()

            output = model(src, trg, src_lengths, trg_lengths, src_mask, tf_ratio=0.0)
            output = output[:, :-1, :].contiguous().view(-1, params.vocab_size)
            trg = trg[:, 1:].contiguous().view(-1)

            assert output.size(0) == trg.size(0)

            loss = criterion(output, trg)
            loss_avg.update(loss.item())
    return loss_avg()

def train_model(epoch_num, model, optimizer, train_iter, params):
    """
    Train the Model for one epoch on the training set
    Arguments:
        epoch_num: epoch number during training
        model: the neural network
        optimizer: optimizer for the parameters of the model
        train_iter: BucketIterator over the training data
        params: hyperparameters for the `model`
    """
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=params.pad_token)
    loss_avg = RunningAverage()
    for index, batch in enumerate(train_iter):
        src, src_lengths = batch.src
        trg, trg_lengths = batch.trg
        src_mask = (src != params.pad_token).unsqueeze(-2)

        if params.cuda:
            src, trg = src.cuda(), trg.cuda()

        output = model(src, trg, src_lengths, trg_lengths, src_mask, tf_ratio=params.teacher_forcing_ratio)
        output = output[:, :-1, :].contiguous().view(-1, params.vocab_size)
        trg = trg[:, 1:].contiguous().view(-1)

        assert output.size(0) == trg.size(0)

        optimizer.zero_grad()
        loss = criterion(output, trg)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())

        if index % 50 == 0 and index != 0:
            logging.info("[%d][loss:%5.2f][pp:%5.2f]" %
                                    (index, loss_avg(), math.exp(loss_avg())))
    return loss_avg()

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

    if "attention" in vars(params):
        logging.info("Running Seq2Seq w/ ({0}) Attention...".format(params.attention))
        encoder = Encoder(src_vocab_size=de_size, embed_size=params.embed_size,
                        hidden_size=params.hidden_size, enc_dropout=params.enc_dropout, 
                        num_layers=params.n_layers_enc)
        decoder = AttentionDecoder(trg_vocab_size=en_size, embed_size=params.embed_size,
                        hidden_size=params.hidden_size, dec_dropout=params.dec_dropout, 
                        attention=params.attention, num_layers=params.n_layers_dec)
    else:
        logging.info("Running regular Seq2Seq model...")
        encoder = Encoder(src_vocab_size=de_size, embed_size=params.embed_size,
                        hidden_size=params.hidden_size, enc_dropout=params.enc_dropout, 
                        num_layers=params.n_layers_enc)
        decoder = Decoder(trg_vocab_size=en_size, embed_size=params.embed_size,
                        hidden_size=params.hidden_size, dec_dropout=params.dec_dropout, 
                        num_layers=params.n_layers_dec)

    model = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    if params.restore_file:
        restore_path = os.path.join(params.model_dir+"/checkpoints/", params.restore_file)
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, optimizer)

    best_val_loss = float('inf')

    logging.info("Starting training for {} epoch(s)".format(params.epochs))
    for epoch in range(params.epochs):
        logging.info("Epoch {}/{}".format(epoch+1, params.epochs))

        # train the model for one epcoh
        epoch_start_time = time.time()
        train_loss_avg = train_model(epoch, model, optimizer, train_iter, params)
        epoch_end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(epoch_start_time, epoch_end_time)
        logging.info(f'Epoch: {epoch+1:02} | Avg Train Loss: {train_loss_avg} | Time: {epoch_mins}m {epoch_secs}s')

        # evaluate the model on the dev set
        val_loss = float('inf')
        # val_loss = evaluate_loss_on_dev(model, dev_iter, params)
        # logging.info("Val loss after {} epochs: {}".format(epoch+1, val_loss))
        is_best = val_loss <= best_val_loss

        # save checkpoint
        save_checkpoint({
            "epoch": epoch+1,
            "state_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict()},
            is_best=is_best,
            checkpoint=params.model_dir+"/checkpoints/")

        # if val_loss < best_val_loss:
        #     logging.info("- Found new lowest loss!")
        #     best_val_loss = val_loss

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
