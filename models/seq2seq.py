import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.attention import DotProductAttention, BahdanauAttention


class Encoder(nn.Module):
    """ GRU Encoder that represents the source sentence with a fixed sized representation """

    def __init__(self, src_vocab_size, embed_size, hidden_size, enc_dropout, num_layers=1):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(src_vocab_size, embed_size)
        self.dropout = nn.Dropout(enc_dropout)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

    def forward(self, batch, src_lengths=None):
        embed = self.dropout(self.embed(batch))
        packed = pack_padded_sequence(embed, src_lengths, batch_first=True)
        output, hidden = self.gru(packed)  # (batch, num_layers*num_directions, hidden_size)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # sum the bidirectional outputs (you could concatenate the two hidden vectors as well)
        # the outputs are from the last layer in the stacked LSTM
        output = (output[:, :, :self.hidden_size] +
                  output[:, :, self.hidden_size:])
        return output, hidden


class AttentionDecoder(nn.Module):
    """ Conditional GRU Decoder w/ Attention """

    def __init__(self, trg_vocab_size, embed_size, hidden_size, dec_dropout, attention, num_layers=1):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = DotProductAttention(hidden_size=hidden_size) if attention == "dot" else BahdanauAttention(hidden_size=hidden_size)
        self.embed = nn.Embedding(trg_vocab_size, embed_size)
        self.dropout = nn.Dropout(dec_dropout)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size * 2, trg_vocab_size)  # concat the attention vector and the regular hidden state

    def forward(self, batch, prev_h, src_mask, encoder_hidden):

        # embed the current input
        batch = batch.unsqueeze(1)  # (batch, 1)
        embed = self.dropout(self.embed(batch))  # (batch, 1, V)

        # calculate attention weights and apply encoder outputs
        projected_keys = self.attention.key_layer(encoder_hidden)  # [batch_size, seq_len, hidden_size]
        query = prev_h[-1].unsqueeze(1)

        context, attention_scores = self.attention(
            query=query, projected_keys=projected_keys,
            values=encoder_hidden, mask=src_mask)

        # concat embeddings and context vectors and input them into the GRU decoder
        contactenated_rnn_input = torch.cat([embed, context], dim=2)
        output, hidden = self.gru(contactenated_rnn_input, prev_h)

        # concat the output of the  GRU decoder and the context vector and feed to linear layer for predictions over trg vocab
        output_context_concatenated = torch.cat([output, context], dim=2)  # [batch_size, 1, hidden_size*2]
        predictions = self.output(output_context_concatenated).squeeze(1)
        return predictions, hidden, attention_scores


class Decoder(nn.Module):
    """ Conditional GRU Decoder """

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

    def forward(self, batch, prev_h, src_mask=None, encoder_hidden=None):
        batch = batch.unsqueeze(1)  # (batch, 1)
        embed = self.dropout(self.embed(batch))  # (batch, 1, V)
        # initialize the Decoder hidden states with the last hidden states from the Encoder
        outputs, hidden = self.gru(embed, prev_h)
        outputs = outputs.squeeze(1)  # (batch, 1 , hidden_size) => (batch, hidden_size)
        predictions = self.output(outputs)  # (batch, hidden_size) => (batch, trg_vocab_size)
        return predictions, hidden, None


class Seq2Seq(nn.Module):
    """ A Base Seq2Seq Model """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, src_lengths, trg_lengths, src_mask, tf_ratio=0.5):
        batch_size, max_seq_len = trg.shape
        trg_vocab_size = self.decoder.trg_vocab_size

        # tensor to store Decoder outputs
        outputs = torch.zeros(max_seq_len, batch_size, trg_vocab_size, device=self.device)

        # last hidden state of the Encoder is used as the initial hidden state of the Decoder
        output, hidden = self.encoder(src, src_lengths)
        hidden = hidden[:self.decoder.num_layers]

        # first input to the decoder is the <sos> tokens
        input = trg[:, 0]

        for t in range(1, max_seq_len):

            # get translated word for the current timestep
            predictions, hidden, _ = self.decoder(batch=input, prev_h=hidden, src_mask=src_mask, encoder_hidden=output)
            
            # save to outputs tensor
            outputs[t - 1] = predictions

            pred = predictions.max(1)[1]  # (batch,1)
            input = trg[:, t] if random.random() < tf_ratio else pred

        return outputs.transpose(0, 1)
