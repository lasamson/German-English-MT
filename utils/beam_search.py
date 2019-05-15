from operator import itemgetter
from queue import PriorityQueue
import torch.nn.functional as F
import torch
import numpy as np
from utils.utils import make_tgt_mask


class Beam():
    '''
    Class for managing the internals of the beam search process.

    This class from OpenNMT's implementation

    Parameters:
        size: the size of the beam
        alpha: length normalization strength
        params: param related to the `model`
    '''

    def __init__(self, size, alpha, params):

        self.size = size
        self._done = False
        self.params = params
        self.alpha = alpha

        # The score for each translation on the beam.
        self.scores = torch.zeros(
            (size,), dtype=torch.float, device=self.params.device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full(
            (size,), self.params.pad_token, dtype=torch.long, device=self.params.device)]
        self.next_ys[0][0] = self.params.sos_index

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        """Update beam status and check if finished or not."""
        num_words = word_prob.size(1)

        # Sum the previous scores (apply length normalization penalty at each step).
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + \
                self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        # this is weird issue with torch.topk
        # need to run it twice to get the correct outputs
        best_scores, best_scores_id = flat_beam_lk.topk(
            self.size, 0, True, True)  # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(
            self.size, 0, True, True)  # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.params.eos_index:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def length_penalty(self, curr_len, alpha=0.0):
        """
        Add a length penalty to the scores of the hypotheses
        See Google Neural Machine Translation System
        """
        return ((5 + curr_len) / 6.0) ** alpha

    def sort_scores(self):
        """Sort the scores."""
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """Get the decoded sequence for the current timestep."""
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.params.sos_index] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


def translate_batch(model, src_enc, src_mask, beam_size, alpha, params, max_seq_len):
    """ 
    Translate all source sequences in a batch.

    This code is heavilty borrowed from OpenNMT. We have adapted the code to fit our needs
    and to our models

    Note: this method only works for Seq2Seq models that use Transformer as the Encoder/Decoder.
    This will not work for GRU Encoders/Decoders. If you want to use GRU, then look at the
    'beam_search_single` method

    Arguments:
        model: the pytorch model (Seq2Seq object)
        src_enc: the encoder output
        src_mask: the mask used on the source sequence
        beam_size: the size of the beam
        alpha: controls the strength of length normalization
        max_seq_len: the maximum length of a sequence
        params: hyperparams related to the `model`
    Returns:
        This method returns the translations/scores of each src sequence in the batch as a tuple
        (translations, scores)
    """

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        """ Indicate the position of an instance in a tensor """
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(
            src_enc, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k]
                           for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(params.device)

        active_src_enc = collect_active_part(
            src_enc, active_inst_idx, n_prev_active_inst, beam_size)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
            active_inst_idx_list)

        return active_src_enc, active_inst_idx_to_position_map

    def beam_decode_step(
            inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm):
        ''' Decode and update beam status, and then return active beam idx '''

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state()
                               for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).to(params.device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(
                1, len_dec_seq + 1, dtype=torch.long, device=params.device)
            dec_partial_pos = dec_partial_pos.unsqueeze(
                0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos

        def predict_word(dec_seq, dec_pos, enc_output, n_active_inst, n_bm):
            # make the target mask
            # [beam_size, 1, seq_len]
            trg_mask = make_tgt_mask(dec_seq, params.pad_token)

            # do one step of the decoder
            dec_output, _ = model.decoder(
                dec_seq, enc_output, src_mask, trg_mask, None, None)

            # Pick the last step: (bh * bm) * d_h
            dec_output = dec_output[:, -1, :]

            word_prob = F.log_softmax(
                model.generator(dec_output), dim=1)
            word_prob = word_prob.view(n_active_inst, n_bm, -1)

            return word_prob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(
                    word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)

        # [seq_len, beam_size * n_inst, vocab_size]
        word_prob = predict_word(
            dec_seq, dec_pos, enc_output, n_active_inst, n_bm)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(
            inst_dec_beams, word_prob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(
                i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    # Repeat
    n_inst, len_s, d_h = src_enc.size()

    # src_en: [n_inst, len_s, d_h] => [n_inst * beam_size, len_s, d_h]
    src_enc = src_enc.repeat(1, beam_size, 1).view(
        n_inst * beam_size, len_s, d_h)

    # prepare Beams
    inst_dec_beams = [Beam(beam_size, alpha, params)
                      for _ in range(n_inst)]

    # Bookkeeping for active or not
    # if you have one sample in your batch:
    # active_inst_idx_list = [0]
    # inst_idx_to_position_map: {0: 0}
    active_inst_idx_list = list(range(n_inst))
    inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
        active_inst_idx_list)

    # Decode
    for len_dec_seq in range(1, max_seq_len + 1):

        # perform one decode step
        active_inst_idx_list = beam_decode_step(
            inst_dec_beams, len_dec_seq, src_enc, inst_idx_to_position_map, beam_size)

        if not active_inst_idx_list:
            break  # all instances have finised their path to <EOS>

        src_enc, inst_idx_to_position_map = collate_active_info(
            src_enc, inst_idx_to_position_map, active_inst_idx_list)

    batch_hyp, batch_scores = collect_hypothesis_and_scores(
        inst_dec_beams, n_best=1)

    return torch.Tensor(batch_hyp[0]), batch_scores


def beam_search_single(model, encoder_final, encoder_outputs, src_mask, beam_size, alpha, params, max_seq_len=100):
    """
    Perform beam search on a **single example** to get the most likely translation for a given source sequence

    Note: this code only works for Seq2Seq models that use the GRU as the Encoder/Decoder. This will
    not work for the Transformer model. If you want to use beam search for the Transformer Model,
    use the 'translate_batch` method

    Arguments:
        model: the pytorch model
        encoder_final: the final hidden representation from the Encoder 
        encoder_outputs: the outputs of the Encoder (this is a hidden state for each token in the src sequence)
        beam_size: the size of the beam
        src_mask: the src sequence mask
        alpha: controls the strenght of lenght normalization
        params: the hyperparams related to the `model`
        max_seq_len: the maximum lenght of the sequence

    Returns:
        The translation for the src sequence as a torch tensor
    """

    decoder_input = torch.ones(1, 1).fill_(
        params.sos_index).type(torch.LongTensor).to(params.device)

    # decode the <s> input as the first input to the decoder
    output, decoder_hidden = model.decode(decoder_input, encoder_outputs,
                                          src_mask, None, encoder_final)

    prob = F.log_softmax(model.generator(output), dim=-1)
    vocab_size = prob.size(-1)

    # get the top K words from the output from the decoder
    topk, indices = torch.topk(prob, beam_size)
    topk = topk.view(-1)
    indices = indices.view(-1)

    # len(list) = beam_size
    tokens_seq = []
    for i in range(beam_size):
        tokens_seq.append(
            [torch.tensor([params.sos_index]).to(params.device), indices[i]])

    # [num_layers, beam_size, decoder_hidden_dim]
    hidden_seq = decoder_hidden.transpose(
        0, 1).repeat(beam_size, 1, 1).transpose(0, 1)

    # logprob_seq = [beam_size]
    logprob_seq = topk

    # [beam_width, seq_len, 2 * encoder_hidden_dim]
    encoder_outputs = encoder_outputs.squeeze(0).repeat(beam_size, 1, 1)

    while True:
        if indices[0].item() == params.eos_index or len(tokens_seq[0]) == max_seq_len:
            return torch.tensor([j.item() for j in tokens_seq[0]])[1:]

        # evaluate the k beams using the decoder
        output, decoder_hidden = model.decode(indices.unsqueeze(1), encoder_outputs,
                                              src_mask, None, encoder_final, hidden_seq.contiguous())

        # [beam_size, 1, vocab_size]
        prob = F.log_softmax(model.generator(output), dim=-1)

        # [beam_size, 1, vocab_size]
        prob += logprob_seq.unsqueeze(1).unsqueeze(1)

        prob = prob.flatten()
        logprob_seq, indices = torch.topk(prob, beam_size)
        logprob_seq = logprob_seq.view(-1)
        indices = indices.view(-1)

        # which beams were choosen as the top k
        beam_chosen = indices // vocab_size
        indices = indices % vocab_size

        temp_seq = []
        for i in range(beam_size):
            seq = tokens_seq[beam_chosen[i].item()][:]
            seq.append(indices[i])
            temp_seq.append(seq)
            hidden_seq[:, i, :] = decoder_hidden[:, beam_chosen[i].item(), :]
        tokens_seq = temp_seq
