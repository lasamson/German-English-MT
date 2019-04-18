""" Perform either greedy decoding or beam search on a trained model """
import argparse
import os
import torch
from utils.data_loader import load_dataset
from models.seq2seq import Encoder, Decoder, AttentionDecoder, Seq2Seq
from models.attention import DotProductAttention, BahdanauAttention
from utils.utils import HyperParams, load_checkpoint, batch_reverse_tokenization, output_decoded_sentences_to_file
from utils.beam_search import beam_decode
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
            src, src_lengths = batch.src
            src_mask = (src != params.pad_token).unsqueeze(-2)

            if params.cuda:
                src = src.cuda()

            # run the src language through the Encoder
            output, hidden = model.encoder(src, src_lengths)
            hidden = hidden[:model.decoder.num_layers]

            # (batch_size,) ==> filled with <s> tokens initially
            decoder_input = torch.LongTensor([[params.sos_index] for _ in range(src.size(0))], device=device).squeeze(1)  # (batch_size)

            # hold the translated sentences
            decoded_batch = torch.zeros((src.size(0), max_len))

            for t in range(max_len):
                predictions, hidden, _ = model.decoder(decoder_input, hidden, src_mask, output)
                pred = predictions.max(1)[1]
                decoded_batch[:, t] = pred
                decoder_input = pred

            tokens = batch_reverse_tokenization(decoded_batch, params.eos_index, params.itos)
            decoded_sentences.extend(tokens)
    return decoded_sentences


def beam_search(model, dev_iter, params, device, beam_width=5, num_sentences=3):
    decoded_sentences = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(dev_iter):
            outputs = []
            src, src_lengths = batch.src
            src_mask = (src != params.pad_token).unsqueeze(-2)

            print("SRC: ", src.size())

            if params.cuda:
                src = src.cuda()

            # run the src langauge through the Encoder
            # output => [l, n, num_directions*hidden_size], hidden => [num_layers*num_directions, n, hidden_size]
            output, hidden = model.encoder(src, src_lengths)
            hidden = hidden[:model.decoder.num_layers]
            translations = beam_decode(model.decoder, src.size(0), hidden, output, params.sos_index, params.eos_index, beam_width, num_sentences, src_mask, device)

            tokens = batch_reverse_tokenization(translations, params.eos_index, params.itos)
            decoded_sentences.extend(tokens)
    return decoded_sentences


def main(params, greedy, beam_size):
    """
    The main function for decoding a trained MT model
    Arguments:
        params: parameters related to the `model` that is being decoded
        greedy: whether or not to do greedy decoding
        beam_size: size of beam if doing beam search
    """
    _, dev_iter, DE, EN = load_dataset(params.data_path, params.min_freq, params.train_batch_size, params.dev_batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("[DE Vocab Size: ]: {}, [EN Vocab Size]: {}".format(de_size, en_size))
    params.pad_token = EN.vocab.stoi["<pad>"]
    params.eos_index = EN.vocab.stoi["</s>"]
    params.sos_index = EN.vocab.stoi["<s>"]
    params.itos = EN.vocab.itos

    device = torch.device("cuda" if params.cuda else "cpu")
    print("Device: {}".format(device))

    if "attention" in vars(params):
        print("Decoding from Seq2Seq w/ ({0}) Attention...".format(params.attention))
        encoder = Encoder(src_vocab_size=de_size, embed_size=params.embed_size,
                          hidden_size=params.hidden_size, enc_dropout=params.enc_dropout,
                          num_layers=params.n_layers_enc)
        decoder = AttentionDecoder(trg_vocab_size=en_size, embed_size=params.embed_size,
                                   hidden_size=params.hidden_size, dec_dropout=params.dec_dropout,
                                   attention=params.attention, num_layers=params.n_layers_dec)
    else:
        print("Decoding from regular Seq2Seq model...")
        encoder = Encoder(src_vocab_size=de_size, embed_size=params.embed_size,
                          hidden_size=params.hidden_size, enc_dropout=params.enc_dropout,
                          num_layers=params.n_layers_enc)
        decoder = Decoder(trg_vocab_size=en_size, embed_size=params.embed_size,
                          hidden_size=params.hidden_size, dec_dropout=params.dec_dropout,
                          num_layers=params.n_layers_dec)

    model = Seq2Seq(encoder, decoder, device).to(device)

    # load the saved model
    model_path = os.path.join(args.model_dir + "/checkpoints/", params.model_file)
    print("Restoring parameters from {}".format(model_path))
    load_checkpoint(model_path, model)

    if greedy:
        print("Doing Greedy Decoding...")
        outputs = greedy_decoding(model, dev_iter, params, 50, device)  # change to dev iter
        output_decoded_sentences_to_file(outputs, params.model_dir, "greedy_outputs.txt")
    if beam_size:
        print("Doing Beam Search...")
        outputs = beam_search(model, dev_iter, params, device, beam_width=beam_size, num_sentences=5)
        output_decoded_sentences_to_file(outputs, params.model_dir, "beam_search_outputs.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Obtain BLEU scores for trained models")
    p.add_argument("-data_path", type=str, help="location of data")
    p.add_argument("-model_dir", type=str, help="Directory containing model")
    p.add_argument("-model_file", type=str, help="Model file (must be contained in the `checkpoints` directory in model_dir)")
    p.add_argument("-greedy", type=bool, default=True, help="greedy decoding on outputs")
    p.add_argument("-beam_size", type=int, default=False, help="Beam Search on outputs")

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
