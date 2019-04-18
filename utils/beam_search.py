from operator import itemgetter
from queue import PriorityQueue
import torch.nn.functional as F
import torch


class BeamSearchNode(object):
    def __init__(self, decoder_hidden, previousNode, wordId, logProb, length):
        '''
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = decoder_hidden
        self.prevNode = previousNode
        self.wordid = wordId
        self.log_p = logProb
        self.leng = length

    def eval(self, alpha=0.7):
        reward = 0
        # Add here a function for shaping a reward
        lp = ((5 + self.leng) / 6) ** alpha
        return self.log_p / lp


def beam_decode(decoder, N, decoder_hiddens, encoder_outputs, sos_index, eos_index, beam_width, num_sentences, src_mask, device):
    decoded_batch = []
    # iterate sentence by sentence

    for idx in range(N):
        decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)

        decoder_input = torch.LongTensor([sos_index], device=device)

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
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == eos_index and n.prevNode != None:  # EOS check
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

                node = BeamSearchNode(decoder_hidden, n, decoded_word, n.log_p + log_p, n.leng + 1)
                score = -node.eval()
                next_nodes.append((score, node))

            for elem in next_nodes:
                queue.put(elem)
            qsize += len(next_nodes) - 1

        full_sentences_sorted = sorted(full_sentences, key=itemgetter(0))

        translation_path = full_sentences_sorted[0][1].prevNode

        utterence = []
        utterence.append(translation_path.wordid)

        while(translation_path.prevNode != None):
            translation_path = translation_path.prevNode
            utterence.append(translation_path.wordid)

        utterence = utterence[::-1]
        utterence = utterence[1:]

        decoded_batch.append(utterence)

    return decoded_batch
