from operator import itemgetter
from queue import PriorityQueue
import torch.nn.functional as F
import torch

class BeamSearchNode(object):
    def __init__(self, decoder_hidden, prev_node, word_id, log_prob, length):
        self.h = decoder_hidden
        self.prev_node = prev_node
        self.word_id = word_id
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=0.7):
        lp = ((5 + self.length) / 6) ** alpha
        return self.log_prob / lp

def beam_decode(decoder, N, decoder_hiddens, encoder_outputs, sos_index, eos_index, beam_width, num_sentences, src_mask, device):
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
        src_mask: mask on the src sequence
        device: torch device
    
    Returns:
        A tensor with word indicies containing the translation output from beam search
    """

    decoder_hidden = decoder_hiddens[:, 0, :].unsqueeze(1)
    encoder_output = encoder_outputs[0, :, :].unsqueeze(0)

    # input token to the beam search process
    decoder_input = torch.LongTensor([sos_index]).to(device)

    # num sentences to generate before terminating
    full_sentences = []

    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
    queue = PriorityQueue()

    queue.put((-node.eval(), node))
    qsize = 1

    while True:
        if qsize > 2000:
            break

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
        predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, src_mask, encoder_output)

        predictions = F.log_softmax(predictions, dim=1)

        log_prob, indexes = torch.topk(predictions, beam_width)
        log_prob = log_prob.view(-1)
        indexes = indexes.view(-1)

        next_nodes = []

        for k in range(beam_width):
            decoded_word = indexes[k].view(-1)
            log_p = log_prob[k].item()

            node = BeamSearchNode(decoder_hidden, n, decoded_word, n.log_prob + log_p, n.length + 1)
            score = -node.eval()
            next_nodes.append((score, node))

        for elem in next_nodes:
            queue.put(elem)

        qsize += len(next_nodes) - 1

    # if beam search blows up and no full translation are found
    # take the top `num_sentenes` from the priority queue
    if len(full_sentences) == 0:
        full_sentences = [queue.get() for _ in range(num_sentences)] 

    print(full_sentences)
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
