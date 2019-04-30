from operator import itemgetter
from queue import PriorityQueue
import torch.nn.functional as F
import torch
from utils.utils import tile
import numpy as np


def beam_search_single(model, encoder_final, encoder_outputs, beam_size, sos_index, eos_index, src_mask, tgt_mask, alpha, device, max_seq_len=100):
    """
    Perform beam search on a single example to get the most likely translation for a given source sequence
    """
    decoder_input = torch.ones(1, 1).fill_(
        sos_index).type(torch.LongTensor).to(device)

    # decode the <s> input as the first input to the decoder
    output, decoder_hidden = model.decode(decoder_input, encoder_outputs,
                                          src_mask, tgt_mask, encoder_final)

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
            [torch.tensor([sos_index]).to(device), indices[i]])

    # [num_layers, beam_size, decoder_hidden_dim]
    hidden_seq = decoder_hidden.transpose(
        0, 1).repeat(beam_size, 1, 1).transpose(0, 1)

    # logprob_seq = [beam_size]
    logprob_seq = topk

    # [beam_width, seq_len, 2 * encoder_hidden_dim]
    encoder_outputs = encoder_outputs.squeeze(0).repeat(beam_size, 1, 1)

    while True:
        if indices[0].item() == eos_index or len(tokens_seq[0]) == max_seq_len:
            return torch.tensor([j.item() for j in tokens_seq[0]])[1:]

        # evaluate the k beams using the decoder
        output, decoder_hidden = model.decode(indices.unsqueeze(1), encoder_outputs,
                                              src_mask, tgt_mask, encoder_final, hidden_seq.contiguous())

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


def beam_search(model, encoder_hidden, encoder_output, sos_index, eos_index, pad_index,
                beam_width, src_mask, tgt_mask, alpha, device, max_len=50):
    """
    Beam Search to find the most likley translation for the given src sequence
    """

    n_best = 1
    batch_size = src_mask.size(0)

    # initialize the hidden state of the decoder
    hidden = model.decoder.init_hidden(encoder_hidden)

   # tile the hidden decoder states and encoder output `beam_size` times
    hidden = tile(hidden, beam_width, dim=1)  # [num_layers, K, hidden_size]

    # encoder_output => [batch_size, seq_len, hidden_size]
    # after tiling => [K, seq_len, hidden_size]
    encoder_output = tile(encoder_output.contiguous(), beam_width, dim=0)

    # src_mask => [batch_size, 1, seq_len]
    # after tile => [K, 1, seq_len]
    src_mask = tile(src_mask, beam_width, dim=0)

    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

    beam_offset = torch.arange(
        0,
        batch_size * beam_width,
        step=beam_width,
        dtype=torch.long,
        device=device
    )

    alive_seq = torch.full(
        [batch_size * beam_width, 1],
        sos_index,
        dtype=torch.long,
        device=device
    )

    # give full probability to the first beam on the first step
    # (batch_size * K)
    topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (beam_width-1),
                                   device=device).repeat(batch_size))

    # structure that holds the finished hypothesis
    hypotheses = [[] for _ in range(batch_size)]

    results = {}
    results["predictions"] = [[] for _ in range(batch_size)]
    results["scores"] = [[] for _ in range(batch_size)]

    # begin beam search
    for step in range(max_len):

        # decoder input
        # alive_seq => [batch_size * K, 1]
        decoder_input = alive_seq[:, -1].view(-1, 1)

        # expand current hypothesis
        # decond one single step
        # ouput is the logits for the final softmax
        pre_output, hidden = model.decoder(
            decoder_input, encoder_output, src_mask, tgt_mask, encoder_hidden, hidden)
        prob = model.generator(pre_output[:, -1])

        # generate predictions
        log_probs = F.log_softmax(prob, dim=1).squeeze(
            1)  # [batch-size*K, trg_vocab_size]

        # [batch_size * K, 1]
        log_probs += topk_log_probs.view(-1).unsqueeze(1)

        curr_scores = log_probs

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step+1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log probs into a list of possiblities
        # [batch_size * K, trg_vocab_size]
        # [batch_size, K * trg_vocab_size]
        curr_scores = curr_scores.reshape(-1,
                                          beam_width * model.decoder.trg_vocab_size)

        # pick currently best top K hypothesis
        # topk_scores: [batch_size, K]
        topk_scores, topk_ids = curr_scores.topk(beam_width, dim=-1)

        # if alpha > -1:
        #     # recover original log probs
        #     topk_log_probs = topk_scores * length_penalty

        # reconstruct the beam origin and true word ids from flattend order
        topk_beam_index = topk_ids.div(model.decoder.trg_vocab_size)
        topk_ids = topk_ids.fmod(model.decoder.trg_vocab_size)

        # map beam_index to batch_index in the flat representation
        batch_index = (
            topk_beam_index
            + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
        select_indices = batch_index.view(-1)

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices),
             topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_len:
            is_finished.fill_(1)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(1)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, beam_width, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = is_finished[i].nonzero().view(-1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
                    hypotheses[b].append((
                        topk_scores[i, j],
                        predictions[i, j, 1:])  # ignore start_token
                    )
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(0).nonzero().view(-1)
            # if all sentences are translated, no need to go further
            # pylint: disable=len-as-condition
            if len(non_finished) == 0:
                break
            # remove finished batches for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished) \
                .view(-1, alive_seq.size(-1))

            # reorder indices, outputs and masks
            select_indices = batch_index.view(-1)
            encoder_output = encoder_output.index_select(0, select_indices)
            src_mask = src_mask.index_select(0, select_indices)

            if isinstance(hidden, tuple):
                # for LSTMs, states are tuples of tensors
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
                # for GRUs, states are single tensors
                hidden = hidden.index_select(1, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                       pad_value=pad_index)

    # TODO also return attention scores and probabilities
    return torch.from_numpy(final_outputs)
