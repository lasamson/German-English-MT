import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils.beam_search import beam_search
from tqdm import tqdm
import logging
import numpy as np
from utils.utils import make_tgt_mask
import os


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
        Perform greedy decoding to obtain translations of the src sequences 
        Arguments:
            max_len: maximum length of target sequence
        """
        decoded_sentences = []
        self.model.eval()
        with tqdm(total=len(self.dev_iter)) as t:
            with torch.no_grad():
                for idx, batch in enumerate(self.dev_iter):
                    src, src_lengths = batch.src
                    trg, trg_lengths = batch.trg
                    src_mask = (src != self.params.pad_token).unsqueeze(-2)

                    # [batch_size, trg_seq_len, trg_seq_len]
                    trg_mask = make_tgt_mask(trg, self.params.pad_token)

                    if self.params.cuda:
                        src = src.cuda()

                    # run the src language through the Encoder
                    encoder_output, encoder_final = self.model.encode(
                        src, src_mask, src_lengths)

                    if self.params.model_type == "GRU":
                        encoder_final = encoder_final[:self.model.decoder.num_layers]
                    else:
                        encoder_final = None

                        # (batch_size,1) ==> filled with <s> tokens initially
                    decoder_input = torch.LongTensor([[self.params.sos_index] for _ in range(
                        src.size(0))]).to(self.device)  # (batch_size,1)

                    # hold the translated sentences
                    # [batch_size, seq_len]
                    decoded_batch = torch.zeros((src.size(0), max_len))

                    hidden = None
                    for i in range(max_len):

                        # predictions: [batch_size, 1, hidden_size], hidden: [num_layers, batch_size, hidden_size]
                        pre_output, hidden = self.model.decoder(
                            decoder_input, encoder_output, src_mask, trg_mask, encoder_final, hidden)

                        # pass the pre_output through the generator to get prediction
                        # pre_ouput[:, -1] => [batch_size, hidden_size]
                        # linear [hidden_size, tgt_vocab_size]
                        # prob: [batch_size, tgt_vocab_size]
                        prob = self.model.generator(pre_output[:, -1])
                        prob = F.log_softmax(prob, dim=-1)

                        # [batch_size, 1]
                        next_word = torch.argmax(prob, dim=-1).unsqueeze(1)

                        decoded_batch[:, i] = next_word

                        decoder_input = next_word

                    tokens = self.batch_reverse_tokenization(decoded_batch)
                    decoded_sentences.extend(tokens)
                    t.update()
        return decoded_sentences

    def beam_decode(self, beam_width=5, num_sentences=3):
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
                    trg, trg_lengths = batch.trg
                    src_mask = (src != self.params.pad_token).unsqueeze(-2)
                    # [batch_size, trg_seq_len, trg_seq_len]
                    trg_mask = make_tgt_mask(trg, self.params.pad_token)

                    if self.params.cuda:
                        src = src.cuda()

                    # run the src langauge through the Encoder
                    # output => [batch_size, seq_len, hidden_size], hidden => [num_layers, batch_size, hidden_size]
                    encoder_output, encoder_final = self.model.encode(
                        src, src_mask, src_lengths)

                    if self.params.model_type == "GRU":
                        encoder_final = encoder_final[:self.model.decoder.num_layers]
                    else:
                        encoder_final = None

                    # output: [batch_size, seq_len, hidden_size], hidden: [num_layers, batch_size, hidden_size]
                    # translations = beam_decode_iterative(self.model, encoder_final, encoder_output, self.params.sos_index, self.params.eos_index, beam_width, num_sentences, src_mask, trg_mask, self.device)
                    # translations = beam_decode(self.model.decoder, hidden, output, self.params.sos_index, self.params.eos_index, beam_width, num_sentences, src_mask, self.device)

                    # run beam search to get translations
                    translations = beam_search(model=self.model, encoder_hidden=encoder_final, encoder_output=encoder_output,
                                               sos_index=self.params.sos_index, eos_index=self.params.eos_index,
                                               pad_index=self.params.pad_token, beam_width=beam_width,
                                               src_mask=src_mask, tgt_mask=trg_mask,
                                               alpha=1, device=self.device, max_len=50)

                    # convert tensor of word indicies to words
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
        filepath = os.path.join(
            self.params.model_dir + "/outputs/", output_file)
        if not os.path.exists(self.params.model_dir + "/outputs"):
            os.mkdir(self.params.model_dir + "/outputs")
        with open(filepath, "w") as f:
            for sentence in outputs:
                sentence = " ".join(sentence)
                f.write(sentence + '\n')
