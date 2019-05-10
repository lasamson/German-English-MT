#!/usr/bin/env python
""" 
Perform either greedy decoding or beam search on a trained model 
and evaluates the translations using BLEU score
"""
import argparse
import subprocess
import os
import torch
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
import logging
from utils.trainer import Trainer
from utils.translator import Translator
from utils.data_loader import load_dataset
from utils.utils import HyperParams
from utils.average_models import average_checkpoints
from models.seq2seq import make_seq2seq_model


def main(params, greedy, beam_size, test):
    """
    The main function for decoding a trained MT model
    Arguments:
        params: parameters related to the `model` that is being decoded
        greedy: whether or not to do greedy decoding
        beam_size: size of beam if doing beam search
    """
    print("Loading dataset...")
    _, dev_iter, test_iterator, DE, EN = load_dataset(
        params.data_path, params.train_batch_size, params.dev_batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("[DE Vocab Size: ]: {}, [EN Vocab Size]: {}".format(de_size, en_size))

    params.src_vocab_size = de_size
    params.tgt_vocab_size = en_size
    params.sos_index = EN.vocab.stoi["<s>"]
    params.pad_token = EN.vocab.stoi["<pad>"]
    params.eos_index = EN.vocab.stoi["</s>"]
    params.itos = EN.vocab.itos

    device = torch.device('cuda' if params.cuda else 'cpu')
    params.device = device

    # make the Seq2Seq model
    model = make_seq2seq_model(params)

    # load the saved model for evaluation
    if params.average > 1:
        print("Averaging the last {} checkpoints".format(params.average))
        checkpoint = {}
        checkpoint["state_dict"] = average_checkpoints(
            params.model_dir, params.average)
        model = Trainer.load_checkpoint(model, checkpoint)
    else:
        model_path = os.path.join(
            params.model_dir + "checkpoints/", params.model_file)
        print("Restoring parameters from {}".format(model_path))
        model = Trainer.load_checkpoint(model, model_path)

    # evaluate on the test set
    if test:
        print("Doing Beam Search on the Test Set")
        test_decoder = Translator(model, test_iterator, params, device)
        test_beam_search_outputs = test_decoder.beam_decode(
            beam_width=beam_size)
        test_decoder.output_decoded_translations(
            test_beam_search_outputs, "beam_search_outputs_size_test={}.en".format(beam_size))
        return

    # instantiate a Translator object to translate SRC langauge to TRG language using Greedy/Beam Decoding
    decoder = Translator(model, dev_iter, params, device)

    if greedy:
        print("Doing Greedy Decoding...")
        greedy_outputs = decoder.greedy_decode(max_len=100)
        decoder.output_decoded_translations(
            greedy_outputs, "greedy_outputs.en")

        print("Evaluating BLEU Score on Greedy Tranlsation...")
        subprocess.call(['./utils/eval.sh', params.model_dir +
                         "outputs/greedy_outputs.en"])

    if beam_size:
        print("Doing Beam Search...")
        beam_search_outputs = decoder.beam_decode(beam_width=beam_size)
        decoder.output_decoded_translations(
            beam_search_outputs, "beam_search_outputs_size={}.en".format(beam_size))

        print("Evaluating BLEU Score on Beam Search Translation")
        subprocess.call(['./utils/eval.sh', params.model_dir +
                         "outputs/beam_search_outputs_size={}.en".format(beam_size)])


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Obtain BLEU scores for trained models")
    p.add_argument("-data_path", type=str, help="location of data")
    p.add_argument("-model_dir", type=str, help="Directory containing model")
    p.add_argument("-model_file", type=str,
                   help="Model file (must be contained in the `checkpoints` directory in model_dir)")
    p.add_argument("-greedy", action="store_true",
                   help="greedy decoding on outputs")
    p.add_argument("-beam_size", type=int, default=5,
                   help="Beam Search on outputs")
    p.add_argument("-average", type=int, default=0,
                   help="Average the weight of the last n checkpoints")
    p.add_argument("-test", action="store_true",
                   help="evaluate on the test set")
    args = p.parse_args()

    json_params_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(
        json_params_path), "No JSON configuration file found at {}".format(json_params_path)
    params = HyperParams(json_params_path)

    params.data_path = args.data_path
    params.model_dir = args.model_dir
    params.model_file = args.model_file
    params.average = args.average
    params.cuda = torch.cuda.is_available()
    main(params, args.greedy, args.beam_size, args.test)
