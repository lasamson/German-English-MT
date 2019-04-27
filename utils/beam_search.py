from operator import itemgetter
from queue import PriorityQueue
import torch.nn.functional as F
import torch
from utils.utils import tile
import numpy as np

class BeamSearchNode(object):
    """ 
    Represents one node in the Beam Search Process 

    Arguments:
        decoder_hidden: hidden state of the decoder at the current timestep
        prev_node: parent node of the current Beam Search Node
        word_id: index of the word
        log_prob: log probability of partial hypothesis
        length: length of partial hypothessis
    """

    def __init__(self, decoder_hidden, prev_node, word_id, log_prob, length):
        self.h = decoder_hidden
        self.prev_node = prev_node
        self.word_id = word_id
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=0.7):
        # alpha = 1.0
        lp = ((5 + self.length) / 6) ** alpha
        return self.log_prob / lp
        # return self.log_prob / float(self.length - 1 + 1e-6) + alpha

    def __lt__(self, other):
        return -self.eval() < -other.eval()


def beam_decode_iterative(model, decoder_hiddens, encoder_outputs, sos_index, eos_index, beam_width, num_sentences, src_mask, tgt_mask, device, max_len=50):
    decoder_hidden = decoder_hiddens[:, 0, :].unsqueeze(
        1)  # [num_layers, 1, hidden_size]
    encoder_output = encoder_outputs[0, :, :].unsqueeze(
        0)  # [1, seq_len, hidden_size]

    # initialize the hidden state for the decoder
    hidden = None

    # input token to the beam search process
    decoder_input = torch.LongTensor([sos_index]).unsqueeze(1).to(device)

    # num sentences to generate before terminating
    full_sentences = []

    # make queue
    queue = PriorityQueue()

    # decode <s>
    pre_output, hidden = model.decoder(
        decoder_input, encoder_output, src_mask, tgt_mask, decoder_hidden, hidden)
    prob = model.generator(pre_output[:, -1])
    predictions = F.log_softmax(prob, dim=-1)

    initial_node = BeamSearchNode(None, None, decoder_input, 0, 1)

    log_prob, indexes = torch.topk(predictions, beam_width)
    log_prob = log_prob.view(-1)
    indexes = indexes.view(-1)

    next_nodes = []
    for k in range(beam_width):
        decoded_word = indexes[k].view(-1)
        log_p = log_prob[k].item()

        node = BeamSearchNode(hidden, initial_node, decoded_word,
                              initial_node.log_prob + log_p, initial_node.length + 1)
        score = -node.eval()
        next_nodes.append((score, node))

    for elem in next_nodes:
        queue.put(elem)

    sent_length = 1
    while True:

        # only decode upto sentences of size `max_len`
        if sent_length >= max_len:
            break

        score_to_node = {}
        for _ in range(beam_width):
            score, n = queue.get()
            decoder_input = n.word_id
            decoder_hidden = n.h

            if n.word_id.item() == eos_index and n.prev_node != None:  # EOS check
                full_sentences.append((score, n))
                if(len(full_sentences) >= num_sentences):  # if we have enough complete sentences
                    break
                else:
                    continue

            # decode for one step
            decoder_input = decoder_input.unsqueeze(1)
            pre_output, hidden = model.decoder(
                decoder_input, encoder_output, src_mask, tgt_mask, decoder_hidden, hidden)
            prob = model.generator(pre_output[:, -1])
            predictions = F.log_softmax(prob, dim=-1)

            # generate predictions
            predictions = F.log_softmax(predictions, dim=1)

            # choose the top K scoring predictions
            log_prob, indexes = torch.topk(predictions, beam_width)
            log_prob = log_prob.view(-1)
            indexes = indexes.view(-1)

            # next_nodes = []

            for k in range(beam_width):
                decoded_word = indexes[k].view(-1)
                log_p = log_prob[k].item()

                node = BeamSearchNode(
                    decoder_hidden, n, decoded_word, n.log_prob + log_p, n.length + 1)
                score = -node.eval()
                # next_nodes.append((score, node))
                score_to_node[score] = node

        # for elem in next_nodes:
        #     queue.put(elem)
        if(len(full_sentences) >= num_sentences):  # if we have enough complete sentences
            print('Full sentences reached')
            break

        ct = 0
        # print('-----')
        for key in sorted(score_to_node):
            # print(key)
            if ct >= beam_width:
                break
            queue.put((key, score_to_node[key]))
            ct += 1

        # qsize += len(next_nodes) - 1
        sent_length += 1

    # if beam search blows up and no full translation are found
    # take the top `num_sentenes` from the priority queue
    if len(full_sentences) == 0:
        full_sentences = [queue.get() for _ in range(num_sentences)]

    full_sentences_sorted = sorted(full_sentences, key=itemgetter(0))
    translation_path = full_sentences_sorted[0][1].prev_node
    utterence = []
    utterence.append(translation_path.word_id)

    while(translation_path.prev_node != None):
        translation_path = translation_path.prev_node
        utterence.append(translation_path.word_id)

    utterence = utterence[::-1]
    utterence = utterence[1:]

    return torch.tensor(utterence).view(1, -1)


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


def beam_decode(model, decoder_hiddens, encoder_outputs, sos_index, eos_index, beam_width, num_sentences, src_mask, tgt_mask, device):
    """
    Perform Beam Search (translation) on a single src sequence 
    Arguments:
        decoder: decoder of the seq2seq model
        decoder_hiddens: initial hidden state of the decoder [num_layers, batch_size, hidden_size]
        encoder_outputs: encoder outputs for a single sequence [batch_size, seq_len, hidden_size]
        sos_index: start of sequence index
        eos_index: end of sequence index
        beam_width: size of beam
        num_sentences: max number of translations for given src sequence
        src_mask: mask on the src sequence [batch_size, seq_len]
        device: torch device

    Returns:
        A tensor with word indicies containing the translation output from beam search
    """

    decoder_hidden = decoder_hiddens[:, 0, :].unsqueeze(1)
    encoder_output = encoder_outputs[0, :, :].unsqueeze(0)

    # input token to the beam search process
    decoder_input = torch.LongTensor([sos_index]).unsqueeze(1).to(device)

    # num sentences to generate before terminating
    full_sentences = []

    hidden = None
    node = BeamSearchNode(hidden, None, decoder_input, 0, 1)
    queue = PriorityQueue()

    queue.put((-node.eval(), node))
    qsize = 1

    while True:
        if qsize > 2000:
            print('Q full')
            break

        score, n = queue.get()
        decoder_input = n.word_id
        decoder_hidden = n.h

        if n.word_id.item() == eos_index and n.prev_node != None:  # EOS check
            full_sentences.append((score, n))
            if(len(full_sentences) >= num_sentences):  # if we have enough complete sentences
                # print('Full sentences reached')
                break
            else:
                continue

        # decode for one step
        pre_output, hidden = model.decoder(
            decoder_input, encoder_output, src_mask, tgt_mask, decoder_hidden, hidden)
        prob = model.generator(pre_output[:, -1])

        # generate predictions
        predictions = F.log_softmax(prob, dim=1)

        # choose the top K scoring predictions
        log_prob, indexes = torch.topk(predictions, beam_width)
        log_prob = log_prob.view(-1)
        indexes = indexes.view(-1)

        next_nodes = []

        for k in range(beam_width):
            decoded_word = indexes[k].view(-1)
            log_p = log_prob[k].item()

            node = BeamSearchNode(
                decoder_hidden, n, decoded_word, n.log_prob + log_p, n.length + 1)
            score = -node.eval()
            next_nodes.append((score, node))

        for elem in next_nodes:
            queue.put(elem)

        qsize += len(next_nodes) - 1

    # if beam search blows up and no full translation are found
    # take the top `num_sentenes` from the priority queue
    if len(full_sentences) == 0:
        full_sentences = [queue.get() for _ in range(num_sentences)]

    full_sentences_sorted = sorted(full_sentences, key=itemgetter(0))
    translation_path = full_sentences_sorted[0][1].prev_node
    utterence = []
    utterence.append(translation_path.word_id)

    while(translation_path.prev_node != None):
        translation_path = translation_path.prev_node
        utterence.append(translation_path.word_id)

    utterence = utterence[::-1]
    utterence = utterence[1:]

    return torch.tensor(utterence).view(1, -1)
