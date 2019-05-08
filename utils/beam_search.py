from operator import itemgetter
from queue import PriorityQueue
import torch.nn.functional as F
import torch
import numpy as np
from utils.beam import Beam
from utils.utils import make_tgt_mask


def translate_batch(model, src_enc, src_mask, beam_size, alpha, params, max_seq_len):
    """ 
    Translate all source sequences in a batch.

    This code is heavilty borrowed from OpenNMT. We have adapted the code to fit our needs
    and our models

    Note: this method only works for Seq2Seq models that use Transformer as the Encoder/Decoder.
    This will not work for GRU Encoders/Decoders. If you want to use GRU, then look at the
    'beam_search_single` method

    Arguments:
        model: the pytorch model (Seq2Seq object)
        src_enc: the encoder output
        src_mask: the mask used on the source sequence
        beam_size: the size of the beam
        alpha: controls the strenght of length normalization
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

        # apply a lenght normalization penalty (look at Google NMT system)
        penalty = ((5 + len(tokens_seq)) / 6.0) ** alpha

        # [beam_size, 1, vocab_size]
        prob += (logprob_seq.unsqueeze(1).unsqueeze(1) / penalty)

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
