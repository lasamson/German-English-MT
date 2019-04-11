import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from models.Seq2SeqAttn import Encoder, Decoder, Seq2SeqAttn
from utils import HyperParams, load_dataset, set_logger, load_checkpoint, save_checkpoint
import os, sys
import logging

def evaluate_model(model, dev_iter, params):
    """
    Evaluate the Model on the Dev Set

    Arguments:
        model: the neural network
        dev_iter: BucketIterator for the dev set
        params: hyperparameters for the `model`
        vocab_size: size of vocab for the target language (EN)
        EN: TorchText Field object of the target language (EN)
    """
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=params.pad_token)
    with torch.no_grad():
        for index, batch in enumerate(dev_iter):
            src, trg = batch.src, batch.trg

            if params.cuda:
                src, trg = src.cuda(), trg.cuda()

            output = model(src, trg, tf_ratio=0.0)
            output = output[:, :-1, :].contiguous().view(-1, params.vocab_size)
            trg = trg[:, 1:].contiguous().view(-1)

            assert output.size(0) == trg.size(0)

            loss = criterion(output, trg)
            total_loss += loss.item()
    return total_loss / len(dev_iter)

def train_model(epoch_num, model, optimizer, train_iter, params):
    """
    Train the Model for one epoch on the training set

    Arguments:
        model: the neural network
        optimizer: optimizer for the parameters of the model
        train_iter: BucketIterator over the training data
        params: hyperparameters for the `model`
    """

    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=params.pad_token)
    for index, batch in enumerate(train_iter):
        src, trg = batch.src, batch.trg
        len_src, len_trg = src.size(0), trg.size(0)

        if params.cuda:
            src, trg = src.cuda(), trg.cuda()

        output = model(src, trg, tf_ratio=params.teacher_forcing_ratio)
        output = output[:, :-1, :].contiguous().view(-1, params.vocab_size)
        trg = trg[:, 1:].contiguous().view(-1)

        assert output.size(0) == trg.size(0)

        optimizer.zero_grad()
        loss = criterion(output, trg)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
        optimizer.step()
        total_loss += loss.item()

        if index % 100 == 0 and index != 0:
            total_loss = total_loss / 100
            logging.info("[%d][loss:%5.2f][pp:%5.2f]" %
                                    (b, total_loss, math.exp(total_loss)))
            total_loss = 0

def main(params, model_dir, restore_file):
    """
    The main function for training the Seq2Seq model with Dot-Product Attention

    Arguments:
        params: hyperparameters for the model
        model_dir: directory of the model
        restore_file: restore file for the model
    """

    logging.info("Loading the datasets...")
    train_iter, dev_iter, DE, EN = load_dataset(params.data_path, params.min_freq, params.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    logging.info("[DE Vocab Size]: {}, [EN Vocab Size]: {}".format(de_size, en_size))
    logging.info("- done.")

    params.vocab_size = en_size
    params.pad_token = EN.vocab.stoi["<pad>"]

    print(EN.vocab.stoi["<BOS>"])
    print(EN.vocab.stoi["<EOS>"])
    # Instantiate the Seq2Seq model
    encoder = Encoder(input_size=de_size, embed_size=params.embed_size,
                      hidden_size=params.hidden_size, num_layers=params.n_layers_enc)
    decoder = Decoder(output_size=en_size, embed_size=params.embed_size,
                      hidden_size=params.hidden_size, num_layers=params.n_layers_dec)
    seq2seq = Seq2SeqAttn(encoder, decoder).cuda() if params.cuda else Seq2SeqAttn(encoder, decoder)

    optimizer = optim.Adam(seq2seq.parameters(), lr=params.lr)

    if restore_file:
        print("Restore File")
        restore_path = os.path.join(args.model_dir+"/checkpoints/", args.restore_file)
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, seq2seq, optimizer)

    # evaluate a trained model on the dev set
    if params.evaluate:
        print("Only evaluating trained model on dev set...")
        val_loss = evaluate_model(seq2seq, dev_iter, params)
        logging.info("Val Loss: {}".format(val_loss))
        return

    best_val_loss = float('inf')

    logging.info("Starting training for {} epoch(s)".format(params.epochs))
    for epoch in range(params.epochs):
        logging.info("Epoch {}/{}".format(epoch+1, params.epochs))

        # train the model for one epcoh
        train_model(epoch, seq2seq, optimizer, train_iter, params)

        # evaluate the model on the dev set
        val_loss = evaluate_model(seq2seq, dev_iter, params)
        logging.info("Val loss after {} epochs: {}".format(epoch+1, val_loss))
        is_best = val_loss <= best_val_loss

        # save checkpoint
        save_checkpoint({
            "epoch": epoch+1,
            "state_dict": seq2seq.state_dict(),
            "optim_dict": optimizer.state_dict()},
            is_best=is_best,
            checkpoint=model_dir+"/checkpoints/")

        if val_loss < best_val_loss:
            logging.info("- Found new lowest loss!")
            best_val_loss = val_loss


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Seq2Seq w/ Attention Hyperparameters")
    p.add_argument("-data_path", type=str, help="location of data")
    p.add_argument("-model_dir", default="./experiments/seq2seq", help="Directory containing seq2seq experiments")
    p.add_argument("-evaluate", default=False, help="Evaluate the model on the dev set. This requires a `restore_file`.")
    p.add_argument("-restore_file", default=None, help="Name of the file in the model directory containing weights \
                   to reload before training")
    args = p.parse_args()

    # Set the logger
    set_logger(os.path.join(args.model_dir, "train.log"))

    json_params_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_params_path), "No json configuration file found at {}".format(json_params_path)
    params = HyperParams(json_params_path)


    params.evaluate = args.evaluate
    params.data_path = args.data_path

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # manual seed for reproducing experiments
    torch.manual_seed(2)
    if params.cuda: torch.cuda.manual_seed(2)

    main(params, args.model_dir, args.restore_file)