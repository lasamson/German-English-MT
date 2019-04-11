import argparse
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
#  from model import Encoder, Decoder, Seq2Seq
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
    for index, batch in enumerate(dev_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.tranpose(0, 1), trg.tranpose(0, 1)

        if params.cuda:
            src = Variable(src.data.cuda(), volatile=True)
            trg = Variable(trg.data.cuda(), volatile=True)
        else:
            src = Variable(src.data, volatile=True)
            trg = Variable(trg.data, volatile=True)

        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = criterion(output[1:].view(-1, params.vocab_size), trg[1:].contiguous().view(-1))
        total_loss += loss
    return total_loss / len(dev_iter)

def train_model(epoch_num, model, optimizer, train_iter, params):
    """
    Train the Model for one epoch on the train set

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
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.tranpose(0, 1), trg.tranpose(0, 1)

        if params.cuda:
            src, trg = src.cuda(), trg.cuda()
        else:
            src, trg = src, trg

        output = model(src, trg, teacher_forcing_ratio=params.teacher_forcing_ratio)

        optimizer.zero_grad()
        loss = criterion(output[1:].view(-1, params.vocab_size), trg[1:].contiguous().view(-1))
        loss.backward()
        clip_grad_norm(model.parameters(), params.grad_clip)
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

    # Instantiate the Seq2Seq model
    #  encoder = Encoder(de_size, params.embed_size, params.hidden_size, n_layers=params.n_layers_enc)
    #  decoder = Decoder(params.embed_size, params.hidden_size, en_size, n_layers=params.n_layers_dec)
    #  seq2seq = Seq2Seq(encoder, decoder).cuda() if params.cuda else Seq2Seq(encoder, decoder)

    optimizer = optim.Adam(seq2seq.parameters(), lr=params.lr)

    if restore_file:
        restore_path = os.path.join(args.model_dir, args.restore_file)
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, seq2seq, checkpoint)

    # evaluate a trained model on the dev set
    if params.evaluate:
        val_loss = evaluate_model(seq2seq, dev_iter, params)
        logging.info("Val Loss: {}".format(val_loss))
        return

    best_val_loss = float('inf')

    logging.info("Starting training for {} epoch(s)".format(params.epochs))
    for epoch in range(params.epochs):
        logging.info("Epoch {}/{}".format(epoch+1, params.num_epochs))

        # train the model for one epcoh
        train_model(epoch, seq2seq, optimizer, train_iter, params)

        # evaluate the model on the dev set
        val_loss = evaluate_model(seq2seq, dev_iter, params)
        logging.info("Val Loss after {} epochs: {}".format(epoch+1, val_loss))

        is_best = val_loss <= best_val_loss

        # save checkpoint
        save_checkpoint({
            "epoch": epoch+1,
            "state_dict": seq2seq.state_dict(),
            "optim_dict": optimizer.state_dict(),
            is_best:is_best,
            checkpoint:model_dir
        })

        if val_loss < best_val_loss:
            logging.info("- Found new lowest loss!")
            best_val_loss = val_loss


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Seq2Seq w/ Attention Hyperparameters")
    p.add_argument("-data_path", type=str, help="location of data")
    p.add_argument("-model_dir", default="./experiments/seq2seq", help="Directory containing seq2seq experiments")
    p.add_argument("--evaluate", default=False, help="Evaluate the model on the dev set. This requires a `restore_file`.")
    p.add_argument("-restore_file", default=None, help="Name of the file in the model directory containing weights \
                   to reload before training")
    args = p.parse_args()

    print("Arguments: {}".format(args))

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
