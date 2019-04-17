import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, hidden_size, enc_dropout, num_layers=2):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(self.src_vocab_size, embed_size)
        self.dropout = nn.Dropout(enc_dropout)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

    def forward(self, batch):
        embedded = self.dropout(self.embed(batch))
        _, hidden = self.gru(embedded) # (batch, num_layers*num_directions, hidden_size)
        return hidden

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, hidden_size, dec_dropout, num_layers=1):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(trg_vocab_size, embed_size)
        self.dropout = nn.Dropout(dec_dropout)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, trg_vocab_size)

    def forward(self, batch, prev_h):
        batch = batch.unsqueeze(1) # (batch, 1)
        embed = self.dropout(self.embed(batch)) # (N, 1, V)
        # initialize the Decoder hidden states with the last hidden states from the Encoder
        outputs, hidden = self.gru(embed, prev_h)
        outputs = outputs.squeeze(1) # (batch, 1 , hidden_size) => (batch, hidden_size)
        predictions = self.output(outputs) # (batch, hidden_size) => (batch, trg_vocab_size)
        return predictions, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, tf_ratio=0.5):
        batch_size, max_seq_len = trg.shape
        trg_vocab_size = self.decoder.trg_vocab_size

        #tensor to store Decoder outputs
        outputs = torch.zeros(max_seq_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the Encoder is used as the initial hidden state of the Decoder
        hidden = self.encoder(src)
        hidden = hidden[:self.decoder.num_layers]

        #first input to the decoder is the <sos> tokens
        input = trg[:,0]

        for t in range(max_seq_len):
            output, hidden = self.decoder(input, hidden)

            # save to outputs tensor
            outputs[t] = output

            pred = output.max(1)[1] # (batch,1)
            input = trg[:,t] if random.random() < tf_ratio else pred

        return outputs.transpose(0, 1)
