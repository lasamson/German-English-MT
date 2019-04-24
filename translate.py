""" Perform either greedy decoding or beam search on a trained model """
import argparse
import os
import torch
from torch.autograd import Variable
from torch import optim
from utils.data_loader import load_dataset
from utils.utils import HyperParams, load_checkpoint
from models.seq2seq import make_seq2seq_model
from utils.beam_search import beam_decode, beam_decode_iterative
from tqdm import tqdm
import logging
from trainer import Trainer
from translator import Translator

def main(params, greedy, beam_size):
    """
    The main function for decoding a trained MT model
    Arguments:
        params: parameters related to the `model` that is being decoded
        greedy: whether or not to do greedy decoding
        beam_size: size of beam if doing beam search
    """
    print("Loading dataset...")
    _, dev_iter, DE, EN = load_dataset(params.data_path, params.min_freq, params.train_batch_size, params.dev_batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("[DE Vocab Size: ]: {}, [EN Vocab Size]: {}".format(de_size, en_size))

    params.src_vocab_size = de_size
    params.tgt_vocab_size = en_size
    params.pad_token = EN.vocab.stoi["<pad>"]
    params.eos_index = EN.vocab.stoi["</s>"]
    params.sos_index = EN.vocab.stoi["<s>"]
    params.itos = EN.vocab.itos

    device = torch.device('cuda' if params.cuda else 'cpu')
    params.device = device

    # make the Seq2Seq model
    model = make_seq2seq_model(params)
    
    # load the saved model
    model_path = os.path.join(args.model_dir + "/checkpoints/", params.model_file)
    print("Restoring parameters from {}".format(model_path))
    model = Trainer.load_checkpoint(model, model_path)

    # instantiate a Translator object to translate SRC langauge using Greedy/Beam Decoding
    decoder = Translator(model, dev_iter, params, device)

    if greedy:
        print("Doing Greedy Decoding...")
        greedy_outputs = decoder.greedy_decode(max_len=50)
        decoder.output_decoded_translations(greedy_outputs, "greedy_outputs.en")

    if beam_size:
        print("Doing Beam Search...")
        beam_search_outputs = decoder.beam_search(beam_width=beam_size, num_sentences=5)
        print(beam_search_outputs)
        decoder.output_decoded_translations(beam_search_outputs, "beam_search_outputs_size={}.en".format(beam_size))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Obtain BLEU scores for trained models")
    p.add_argument("-data_path", type=str, help="location of data")
    p.add_argument("-model_dir", type=str, help="Directory containing model")
    p.add_argument("-model_file", type=str, help="Model file (must be contained in the `checkpoints` directory in model_dir)")
    p.add_argument("-greedy", action="store_true", help="greedy decoding on outputs")
    p.add_argument("-beam_size", type=int, default=5, help="Beam Search on outputs")

    args = p.parse_args()

    json_params_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_params_path), "No JSON configuration file found at {}".format(json_params_path)
    params = HyperParams(json_params_path)

    params.data_path = args.data_path
    params.model_dir = args.model_dir
    params.model_file = args.model_file
    params.cuda = torch.cuda.is_available()
    main(params, args.greedy, args.beam_size)
