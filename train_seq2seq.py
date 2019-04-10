import argparse
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
#  from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset

def evaluate_model(model, dev_iter, vocab_size, EN):
    """ Evaluate the Model on the Dev Set """
    model.eval()
    pad = EN.vocab.stoi["<pad>"]
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad)
    for index, batch in enumerate(dev_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.tranpose(0, 1), trg.tranpose(0, 1)

        src = Variable(src.data.cuda(), volatile=True)
        trg = Variable(trg.data.cuda(), volatile=True)

        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = criterion(output[1:].view(-1, vocab_size), trg[1:].contiguous().view(-1))
        total_loss += loss
    return total_loss / len(dev_iter)

def train_model(epoch_num, model, optimizer, train_iter, vocab_size, grad_clip, EN):
    """ Train the Model for one epoch on the train set"""
    model.train()
    total_loss = 0
    pad = EN.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad)
    for index, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.tranpose(0, 1), trg.tranpose(0, 1)
        src, trg = src.cuda(), trg.cuda()

        output = model(src, trg, teacher_forcing_ratio)

        optimizer.zero_grad()
        loss = criterion(output[1:].view(-1, vocab_size), trg[1:].contiguous().view(-1))
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

        if index % 100 == 0 and index != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                                    (b, total_loss, math.exp(total_loss)))
            total_loss = 0

def main(args):
    args

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Seq2Seq w/ Attention Hyperparameters")
    p.add_argument("-epochs", type=int, default=100, help="number of epochs for train")
    p.add_argument("-batch_size", type=int, default=128, help="batch size for training")
    p.add_argument("-lr", type=float, default=.0001, help="initial learning rate")
    p.add_argument("-grad_clip", type=float, default=10.0, help="deal with gradient explosion")
    p.add_argument("-min_freq", type=int, default=2, help="min freq to add to vocab for src and trg languages")
    p.add_argument("-embed_size", type=int, default=512, help="size of embeddings for src and trg languages")
    p.add_argument("-hidden_size", type=int, default=512, help="size of the hidden sizes for the Encoder/Decoder RNN")
    p.add_argument("-n_layers_enc", type=int, default=1, help="number of layers for the encoder")
    p.add_argument("-n_layers_dec", type=int, default=1, help="number of layers for the decoder")
    p.add_argument("-teacher_forcing_ratio", type=float, default=.5, help="probability of doing teacher-forcing on each output of the decoder")
    args = p.parse_args()
    main()
