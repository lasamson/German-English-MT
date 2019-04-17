""" Perform either greedy decoding or beam search on a trained model """

import argparse
import os
import torch
from utils.data_loader import load_dataset
from utils.utils import HyperParams, load_checkpoint, batch_reverse_tokenization, output_decoded_sentences_to_file
from models.seq2seq import Encoder, Decoder, Seq2Seq

def greedy_decoding(model, dev_iter, params, max_len, device):
    """ 
    Perform greedy decoding a trained model

    Arguments:
        model: trained Seq2Seq model
        dev_iter: BucketIterator for the Dev Set
        params: parameters related to the `model`
        max_len: maximum length of target sequence
        device: cpu or gpu 
    """

    decoded_sentences = []
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(dev_iter):
            src, _ = batch.src
            if params.cuda:
                src = src.cuda()
            hidden = model.encoder(src)
            hidden = hidden[:model.decoder.num_layers]

            decoder_input = torch.LongTensor([[params.sos_index] for _ in range(src.size(0))], device=device).squeeze(1)  # (batch_size)
            decoded_batch = torch.zeros((src.size(0), max_len))
            
            for t in range(max_len):
                output, hidden = model.decoder(decoder_input, hidden)
                pred = output.max(1)[1]
                decoded_batch[:, t] = pred
                decoder_input = pred

            tokens = batch_reverse_tokenization(decoded_batch, params.eos_index, params.itos)
            print(tokens)
            decoded_sentences.extend(tokens)
    return decoded_sentences

def beam_search():
    pass

def main(params, greedy, beam_size):
    """
    The main function for decoding a trained MT model

    Arguments:
        params: parameters related to the `model` that is being decoded
        greedy: whether or not to do greedy decoding
        beam_size: size of beam if doing beam search
    """
    _, dev_iter, DE, EN = load_dataset(params.data_path, params.min_freq, params.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    params.pad_token = EN.vocab.stoi["<pad>"]
    params.eos_index = EN.vocab.stoi["</s>"]
    params.sos_index = EN.vocab.stoi["<s>"]
    params.itos = EN.vocab.itos

    # instantiate the Seq2Seq model
    encoder = Encoder(src_vocab_size=de_size, embed_size=params.embed_size,
                      hidden_size=params.hidden_size, enc_dropout=params.enc_dropout, num_layers=params.n_layers_enc)
    decoder = Decoder(trg_vocab_size=en_size, embed_size=params.embed_size,
                      hidden_size=params.hidden_size, dec_dropout=params.dec_dropout, num_layers=params.n_layers_dec)
    device = torch.device("cuda" if params.cuda else "cpu")
    seq2seq = Seq2Seq(encoder, decoder, device).to(device)

    model_path = os.path.join(args.model_dir+"/checkpoints/", params.model_file)
    print("Restoring parameters from {}".format(model_path))
    load_checkpoint(model_path, seq2seq)

    if greedy:
        outputs = greedy_decoding(seq2seq, dev_iter, params, 50, device) # change to dev iter
        output_decoded_sentences_to_file(outputs, params.model_dir, "greedy_outputs.txt")

    if beam_size:
        pass

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Obtain BLEU scores for trained models")
    p.add_argument("-data_path", type=str, help="location of data")
    p.add_argument("-model_dir", type=str, help="Directory containing model")
    p.add_argument("-model_file", type=str, help="Model file (must be contained in the `checkpoints` directory in model_dir)")
    p.add_argument("-greedy", type=bool, default=True, help="greedy decoding on outputs")
    p.add_argument("-beam_size", default=False, help="Beam Search on outputs")

    args = p.parse_args()

    print(args)

    json_params_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_params_path), "No JSON configuration file found at {}".format(json_params_path)
    params = HyperParams(json_params_path)

    params.data_path = args.data_path
    params.model_dir = args.model_dir
    params.model_file = args.model_file
    params.cuda = torch.cuda.is_available()

    main(params, args.greedy, args.beam_size)

