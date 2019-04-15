""" Perform either greedy decoding or beam search on a trained model """

import argparse
import os
import torch
from utils.data_loader import load_dataset
from utils.utils import HyperParams, load_checkpoint
from models.lstm_seq2seq import Encoder, Decoder, Seq2Seq


def output_decoded_sentences_to_file(outputs, model_dir, filename):
    """
    Output the decoded sentences to a file

    Arguments:
        outputs: list of decoded sentences from the model
        model_dir: directory of the `model`
        filename: name of the file to output the decoded sentences
    """

    filepath = os.path.join(model_dir + "/outputs/", filename)

    if not os.path.exists(model_dir + "/outputs"):
        os.mkdir(model_dir + "/outputs")

    with open(filepath, "w") as f:
        for sentence in outputs:
            sentence = " ".join(sentence)
            f.write(sentence + "\n")


def batch_reverse_tokenization(batch, params):
    """
    Converts the token IDs to actual words in a batch

    Arguments:
        batch: a tensor of containing the decoded examples (with word ids in the cells)
        params: params of the `model`
    """
    sentences = []
    for example in batch:
        sentence = [params.itos[example[i]] for i in range(batch.size(1))]
        sentences.append(sentence)
    return sentences


def greedy_decoding(model, dev_iter, params):
    """ Do greedy decoding a trained model

    Arguments:
        model: trained Seq2Seq model
        dev_iter: BucketIterator for the Dev Set
        params: parameters related to the `model`
    """

    decoded_sentences = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(dev_iter):
            src, trg = batch.src, batch.trg
            print(src.size(), trg.size())

            if params.cuda:
                src, trg = src.cuda(), trg.cuda()

            output = model(src, trg, tf_ratio=0.0)
            _, output = torch.max(output, 2)
            print(output.size())
            tokens = batch_reverse_tokenization(output, params)
            decoded_sentences.extend(tokens)

    print(len(decoded_sentences))
    print(decoded_sentences[0:2])
    return decoded_sentences


def beam_search(model, dev_iter, params, beam_width=5):
    decoded_sentences = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(dev_iter):
            outputs = []
            src, trg = batch.src, batch.trg
            print(src.size(), trg.size())

            if params.cuda:
                src, trg = src.cuda(), trg.cuda()

            for i, sent in range(len(batch)):
                pass
                output = model(src[i], trg[i], tf_ratio=0.0)
                print(output.size())
                outputs.append(output)
            tokens = batch_reverse_tokenization(outputs, params)
            decoded_sentences.extend(tokens)

    print(len(decoded_sentences))
    print(decoded_sentences[0:2])
    return decoded_sentences


def main(data_path, greedy, beam_size):
    """
    The main function for decoding a trained MT model

    Arguments:
        params: parameters related to the `model` that is being decoded
        greedy: whether or not to do greedy decoding
        beam_size: size of beam if doing beam search
    """

    train_iter, dev_iter, DE, EN = load_dataset(params.data_path, params.min_freq, params.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    params.vocab_size = en_size
    params.pad_token = EN.vocab.stoi["<pad>"]
    params.itos = EN.vocab.itos

    # instantiate the Seq2Seq model
    encoder = Encoder(input_size=de_size, embed_size=params.embed_size,
                      hidden_size=params.hidden_size, num_layers=params.n_layers_enc, dropout=params.dropout_enc)
    decoder = Decoder(output_size=en_size, embed_size=params.embed_size,
                      hidden_size=params.hidden_size, num_layers=params.n_layers_dec, dropout=params.dropout_dec)
    device = torch.device('cuda' if params.cuda else 'cpu')
    seq2seq = Seq2Seq(encoder, decoder, device).to(device)

    model_path = os.path.join(args.model_dir + "/checkpoints/", params.model_file)
    print("Restoring parameters from {}".format(model_path))
    load_checkpoint(model_path, seq2seq)

    if greedy:
        outputs = greedy_decoding(seq2seq, dev_iter, params)
    else:
        outputs = beam_search(seq2seq, dev_iter, params)

    output_decoded_sentences_to_file(outputs, params.model_dir, "greedy_outputs.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Obtain BLEU scores for trained models")
    p.add_argument("-data_path", type=str, help="location of data")
    p.add_argument("-model_dir", type=str, help="Directory containing model")
    p.add_argument("-model_file", type=str, help="Model file")
    p.add_argument("-greedy", type=bool, default=True, help="Greedy Decoding on outputs")
    p.add_argument("-beam_size", default=1, help="Beam Search on outputs")

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
