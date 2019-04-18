import random
import torch
from torch import nn
from utils.utils import BeamSearchNode


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.input_size = input_size # source vocab size
        self.embed_size = embed_size # embed size
        self.hidden_size = hidden_size # hidden size
        self.num_layers = num_layers # num layers
        self.dropout = dropout # encoder dropout prob

        self.dropout = nn.Dropout(dropout) 
        self.embed = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, src, h=None, c=None):
        # src = seq_len, batch_size
        embedded = self.embed(src) # seq_len, batch_size, embed_size
        embedded = self.dropout(embedded) # seq_len, batch_size, embed_size

        # outputs = seq_len, batch_size, num_dirs * hidden_size
        # h, c = num_layers * num_dirs, batch_size, hidden_size
        if (h and c):
            outputs, (h, c) = self.lstm(embedded, (h, c))
        else:
            outputs, (h, c) = self.lstm(embedded)

        return outputs, (h, c)


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.output_size = output_size # target vocab size
        self.embed_size = embed_size # embed size
        self.hidden_size = hidden_size # hidden size
        self.num_layers = num_layers # num layers
        self.dropout = dropout # decoder dropout prob

        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
        self.output = nn.Linear(hidden_size, output_size) # projection layer for prediction

    def forward(self, input, prev_h, prev_c, encoder_outputs):
        # input = batch_size
        input = input.unsqueeze(0) # 1, batch_size
        embedded = self.embed(input) # 1, batch_size, embed_size
        embedded = self.dropout(embedded) # 1, batch_size, embed_size

        # output = 1, batch_size, 1 * hidden_size
        # h, c = 1 * num_layers, batch_size, hidden_size
        output, (h, c) = self.lstm(embedded, (prev_h, prev_c))
        pred = self.output(output.squeeze(0)) # batch_size, output_size

        return pred, (h, c)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, beam_size=1, tf_ratio=0.5):
        seq_len = trg.shape[0] # decoder seq_len
        batch_size = trg.shape[1] # batch_size
        vocab_size = self.decoder.output_size # decoder vocab_size

        outputs = torch.zeros(seq_len, batch_size, vocab_size).to(self.device) # tensor to hold predictions

        encoder_outputs, (h, c) = self.encoder(src)

        input = trg[0] # get first word across all batches, (batch_size, vocab_size)

        if beam_size > 1:
            queue = PriorityQueue()
            qsize = 1

            logp = 0
            node = BeamSearchNode(None, input, logp, 1)
            queue.put((-node.eval(), node))

            endnode = None

            while True:
                if qsize > 2000:
                    break
                score, node = queue.get()
                if node.wordid.item() == 3:  # Check for EOS
                    endnode = node
                    break

                output, (h, c) = self.decoder(node.wordid.item(), h, c, encoder_outputs)
                log_prob, indices = torch.topk(output, beam_size)
                next_nodes = []

                for new_k in range(beam_size):
                    decoded_t = indices[new_k]
                    logp = log_prob[new_k]

                    n = BeamSearchNode(node, decoded_t, node.log_p + logp, node.leng + 1)
                    next_nodes.appemd((-n.eval(), n))

                for i in range(len(next_nodes)):
                    queue.put(next_nodes[i])

                qsize += len(next_nodes) - 1

            if endnode == None:
                endnode = node.get()

            utterance = []
            node = endnode
            utterance.append(endnode.wordid)
            while node.prevNode != None:
                node = node.prevNode
                utterance.append(node.wordid)
            utterance = utterance[::-1]
            outputs = utterance

        else:
            for t in range(1, seq_len):
                # output = batch_size, vocab_size
                # h, c = 1 * num_layers, batch_size, hidden_size
                output, (h, c) = self.decoder(input, h, c, encoder_outputs)
                outputs[t] = output

                pred = output.max(1)[1] # get max along vocab_size and the indices of the max values
                input = trg[t] if random.random() < tf_ratio else pred # teacher forcing

        return outputs
