import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils.beam_search import beam_search_single, translate_batch
from tqdm import tqdm
import logging
import numpy as np
from utils.utils import make_tgt_mask
import os


class Translator(object):
    """
    Translator class that handles Greedy Decoding and Beam Search inorder to obtain translations from the model

    Arguments:  
        model: the trained model to peform decoding on 
        dev_iter: Iterator for the Dev data 
        params: params related to the `model`
        device: CPU/GPU device number
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
                    src_mask = (src != self.params.pad_token).unsqueeze(-2)

                    if self.params.cuda:
                        src = src.cuda()

                    # run the src language through the Encoder
                    encoder_output, encoder_final = self.model.encode(
                        src, src_mask, src_lengths)

                    # TODO: Greedy Decoding for GRU and Transformers are different
                    # the GRU greedy decoding just takes in the previous token
                    # whereas the Transformer model takes in the whole sequence
                    # that has been decoded sofar

                    # Encoder Final is the final hidden state of the Encoder Model
                    # You will only have the Encoder Final if you are using a
                    # GRUEncoder and if you are using a Transformer then
                    # the encoder_final will be None
                    # [num_layers, batch_size, hidden_size]
                    encoder_final = encoder_final[:self.model.decoder.num_layers] if self.params.model_type == "GRU" else None

                    decoded_batch = torch.ones(1, 1).fill_(
                        self.params.sos_index).type_as(src)

                    hidden = None
                    for _ in range(max_len-1):
                        # either use the decoded batch to decode the next word
                        # or use the last word decoded to decode the next work
                        trg = decoded_batch[:, -1].unsqueeze(
                            1) if self.params.model_type == "GRU" else decoded_batch

                        # create trgt_mask for transformer [batch_size, seq_len, seq_len]
                        trg_mask = make_tgt_mask(
                            trg, tgt_pad=self.params.pad_token)

                        # pre_output: [batch_size, seq_len, hidden_size], hidden: [num_layers, batch_size, hidden_size]
                        pre_output, hidden = self.model.decode(
                            trg, encoder_output, src_mask, trg_mask, encoder_final, hidden)

                        # pass the pre_output through the generator to get prediction
                        # pre_ouput[:, -1] => [batch_size, hidden_size]
                        # linear [hidden_size, tgt_vocab_size]
                        # prob: [batch_size, tgt_vocab_size]
                        prob = self.model.generator(pre_output[:, -1])
                        prob = F.log_softmax(prob, dim=-1)

                        # [batch_size, 1]
                        next_word = torch.argmax(prob, dim=-1).item()

                        decoded_batch = torch.cat([decoded_batch,
                                                   torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

                    # the decoded batch should not include the <s> token
                    decoded_batch = decoded_batch[:, 1:]
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
                    src, src_lengths = batch.src
                    src_mask = (src != self.params.pad_token).unsqueeze(-2)

                    if self.params.cuda:
                        src = src.cuda()

                    # run the src langauge through the Encoder
                    # output => [batch_size, seq_len, hidden_size],
                    # hidden => [num_layers, batch_size, hidden_size]
                    encoder_output, encoder_final = self.model.encode(
                        src, src_mask, src_lengths)

                    encoder_final = encoder_final[:self.model.decoder.num_layers] if self.params.model_type == "GRU" else None

                    if self.params.model_type == "GRU":
                        trg_mask = None
                        translation = beam_search_single(model=self.model, encoder_final=encoder_final, encoder_outputs=encoder_output,
                                                         src_mask=src_mask, beam_size=beam_width,
                                                         alpha=1.0, params=self.params, max_seq_len=100)
                    else:
                        translation, _ = translate_batch(model=self.model, src_enc=encoder_output, src_mask=src_mask,
                                                         beam_size=beam_width, alpha=0.0, params=self.params, max_seq_len=100)

                    # convert tensor of word indices to words
                    tokens = self.batch_reverse_tokenization(
                        translation.view(1, -1))
                    print(tokens)
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
