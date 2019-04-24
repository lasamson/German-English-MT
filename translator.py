import torch
from torch.autograd import Variable
from utils.beam_search import beam_decode, beam_decode_iterative
from tqdm import tqdm
import logging
from translator import Translator


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
