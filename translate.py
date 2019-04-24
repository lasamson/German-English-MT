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

class Translator(object):
    """
    Translator class that handles Greedy Decoding and Beam Search inorder to obtain translations from the model
    """

    def __init__(self, model, dev_iter, params, device):
        self.model = model
        self.dev_iter = dev_iter
        self.params = params
        self.device = device

    def greedy_decode(self, max_len):
        """ 
        Perform greedy decoding a trained model
        Arguments:
            max_len: maximum length of target sequence
        """
        decoded_sentences = []
        self.model.eval()
        with tqdm(total=len(self.dev_iter)) as t:
            with torch.no_grad():
                for idx, batch in enumerate(self.dev_iter):
                    src, src_lengths = batch.src
                    src_mask = (src != self.params.pad_token).unsqueeze(-2)

                    if self.params.cuda:
                        src = src.cuda()

                    # run the src language through the Encoder
                    output, hidden = self.model.encoder(src, src_lengths)
                    hidden = hidden[:self.model.decoder.num_layers]

                    # (batch_size,) ==> filled with <s> tokens initially
                    decoder_input = torch.LongTensor([[self.params.sos_index] for _ in range(src.size(0))]).squeeze(1).to(self.device)  # (batch_size)

                    # hold the translated sentences
                    decoded_batch = torch.zeros((src.size(0), max_len))

                    for i in range(max_len):
                        predictions, hidden, _ = self.model.decoder(decoder_input, hidden, src_mask, output)
                        pred = predictions.max(1)[1]
                        decoded_batch[:, i] = pred
                        decoder_input = pred

                    tokens = self.batch_reverse_tokenization(decoded_batch)
                    decoded_sentences.extend(tokens)
                    t.update()
        return decoded_sentences

    def beam_search(self, beam_width=5, num_sentences=3):
        """ 
        Perform Beam Search as a decoding procedure to get translations
        Arguments:
            beam_width: size of the beam
            num_sentences: max number of `hypothesis` to complete before ending beam search
        """
        decoded_sentences = []
        self.model.eval()
        with tqdm(total=len(self.dev_iter)) as t:
            with torch.no_grad():
                for index, batch in enumerate(self.dev_iter):
                    # if (index + 1) % 20 == 0:
                        # break
                    src, src_lengths = batch.src
                    src_mask = (src != self.params.pad_token).unsqueeze(-2)

                    if params.cuda:
                        src = src.cuda()

                    # run the src langauge through the Encoder
                    # output => [l, n, num_directions*hidden_size], hidden => [num_layers, n, hidden_size]
                    output, hidden = self.model.encoder(src, src_lengths)
                    hidden = hidden[:self.model.decoder.num_layers]
                    # output: [batch_size, seq_len, hidden_size], hidden: [num_layers, batch_size, hidden_size]
                    translations = beam_decode_iterative(self.model.decoder, hidden, output, self.params.sos_index, self.params.eos_index, beam_width, num_sentences, src_mask, self.device)
                    # translations = beam_decode(self.model.decoder, hidden, output, self.params.sos_index, self.params.eos_index, beam_width, num_sentences, src_mask, self.device)
                    tokens = self.batch_reverse_tokenization(translations)
                    decoded_sentences.extend(tokens)
                    t.update()
        return decoded_sentences

    def batch_reverse_tokenization(self, batch):
        """
        Convert a batch of sequences of word IDs to words in a batch
        Arguments:
            batch: a tensor containg the decoded examples (with word ids representing the sequence)
        """
        sentences = []
        for example in batch:
            sentence = []
            for token_id in example:
                token_id = int(token_id.item())
                if token_id == self.params.eos_index:
                    break
                sentence.append(self.params.itos[token_id])
            sentences.append(sentence)
        return sentences

    def output_decoded_translations(self, outputs, output_file):
        """
        Outputs a list of decoded translations to an output file
        Arguments:
            outputs: list of decoded sentences from the model (translations)
            modeL_dir: directory of the `model`
            output_file: name of the file to output the translations
        """
        filepath = os.path.join(self.params.model_dir + "/outputs/", output_file)
        if not os.path.exists(self.params.model_dir + "/outputs"):
            os.mkdir(self.params.model_dir + "/outputs")
        with open(filepath, "w") as f:
            for sentence in outputs:
                sentence = " ".join(sentence)
                f.write(sentence + '\n')


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
       
    # make the Seq2Seq model
    model = make_seq2seq_model(params)

    if params.restore_file:
        restore_path = os.path.join(params.model_dir+"/checkpoints/", params.restore_file)
        logging.info("Restoring parameters from {}".format(restore_path))
        trainer.load_checkpoint(restore_path)

    # load the saved model
    model_path = os.path.join(args.model_dir + "/checkpoints/", params.model_file)
    print("Restoring parameters from {}".format(model_path))
    load_checkpoint(model_path, model)

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
