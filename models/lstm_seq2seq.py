import random
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, src, h=None, c=None):
        embedded = self.embed(src)
        embedded = self.dropout(embedded)
        
        if (h and c):
            outputs, (h, c) = self.lstm(embedded, (h, c))
        else:
            outputs, (h, c) = self.lstm(embedded)

        return outputs, (h, c)


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input, prev_h, prev_c, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.embed(input)
        embedded = self.dropout(embedded)

        output, (h, c) = self.lstm(embedded, (prev_h, prev_c))
        pred = self.output(output.squeeze(0))

        return pred, (h, c)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, tf_ratio=0.5):
        seq_len = trg.shape[0]
        batch_size = trg.shape[1]
        vocab_size = self.decoder.output_size

        outputs = torch.zeros(seq_len, batch_size, vocab_size).to(self.device)

        encoder_outputs, (h, c) = self.encoder(src)

        input = trg[0]

        for t in range(1, seq_len):
            output, (h, c) = self.decoder(input, h, c, encoder_outputs)
            outputs[t] = output

            pred = output.max(1)[1]
            input = trg[t] if random.random() < tf_ratio else pred

        return outputs
