import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.attention import DotProductAttention, BahdanauAttention

class Encoder(nn.Module):
    """ GRU Encoder that represents the source sentence with a fixed sized representation """
    def __init__(self, src_vocab_size, embed_size, hidden_size, input_dropout_p, dropout_p, num_layers=1):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(src_vocab_size, embed_size)
        self.dropout = nn.Dropout(input_dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout_p, bidirectional=True, batch_first=True)

    def forward(self, batch, src_lengths=None):
        embed = self.dropout(self.embed(batch))
        packed = pack_padded_sequence(embed, src_lengths, batch_first=True)
        
        # output => [batch_size, seq_len, num_directions * hidden_size]
        # hidden => [num_layers * num_directions, batch_size, hidden_size]
        output, hidden = self.gru(packed) 
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        # the outputs are from the last layer in the stacked LSTM
        fwd_hidden = hidden[0:hidden.size(0):2]
        bwd_hidden = hidden[1:hidden.size(0):2]
        hidden = torch.cat([fwd_hidden, bwd_hidden], dim=2)  # [num_layers, batch, 2*hidden_size]

        return output, hidden


class AttentionDecoder(nn.Module):
    """ Conditional GRU Decoder w/ Attention """
    def __init__(self, trg_vocab_size, embed_size, hidden_size, input_dropout_p, dropout_p, attention, num_layers=1):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = DotProductAttention(hidden_size=hidden_size) if attention == "dot" else BahdanauAttention(hidden_size=hidden_size)
        self.embed = nn.Embedding(trg_vocab_size, embed_size)
        self.dropout = nn.Dropout(input_dropout_p)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, num_layers, dropout=dropout_p, batch_first=True)
        self.output = nn.Linear(hidden_size + hidden_size * 2 + embed_size, trg_vocab_size)  # concat the attention vector and the regular hidden state

    def forward(self, batch, prev_h, context, src_mask, encoder_hidden):
        # embed the current input
        batch = batch.unsqueeze(1)  # [batch, 1]
        embed = self.dropout(self.embed(batch))  # [batch, 1, V]

        # calculate attention weights and apply encoder outputs
        # encoder_hidden => [batch_size, seq_len, hidden_size*2]
        projected_keys = self.attention.key_layer(encoder_hidden)  # [batch_size, seq_len, hidden_size]

        # get the top most layer of the decoder GRU as the hidden representation
        # prev_h => [num_layers, batch_size, hidden_size]
        # query => [batch_size, 1, hidden_size]
        query = prev_h[-1].unsqueeze(1) 

        # context => [batch_size, 1, hidden_size]
        # attention_scores => [batch_size, 1, hidden_size]
        context, attention_scores = self.attention(
            query=query, projected_keys=projected_keys,
            values=encoder_hidden, mask=src_mask)

        # concat embeddings and context vectors and input them into the GRU decoder
        contactenated_rnn_input = torch.cat([embed, context], dim=2)
        output, hidden = self.gru(contactenated_rnn_input, prev_h)

        # concat the output of the  GRU decoder and the context vector and feed to linear layer for predictions over trg vocab
        # output: [batch_size,1, hidden_size] embed: [batch_size, 1, hidden_size] context: [batch_size, 1, hidden_size]
        output_context_concatenated = torch.cat([embed, output, context], dim=2)  # [batch_size, 1, hidden_size + hidden_size*2 + embed_size]

        # predictions => [batch_size, trg_vocab_size]
        predictions = self.output(output_context_concatenated).squeeze(1)

        return predictions, hidden, attention_scores


class Decoder(nn.Module):
    """ 
    Conditional GRU Decoder 
    """
    def __init__(self, trg_vocab_size, embed_size, hidden_size, input_dropout_p, dropout_p, num_layers=1):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(trg_vocab_size, embed_size)
        self.dropout = nn.Dropout(input_dropout_p)
        self.gru = nn.GRU(embed_size + 2 * hidden_size, hidden_size, num_layers, dropout=dropout_p, batch_first=True)
        self.output = nn.Linear(embed_size + 2 * hidden_size + hidden_size, trg_vocab_size)

    def forward(self, batch, prev_h, context, src_mask=None, encoder_hidden=None):
        batch = batch.unsqueeze(1)  # [batch, 1]
        embed = self.dropout(self.embed(batch))  # [batch, 1, V]

        # concatenate the embedding + context vector
        # context => [batch_size, 1, hidden_size * 2]
        embed_con = torch.cat((embed, context), dim = 2) # [batch, 1, V + 2 * hidden_size]

        # initialize the Decoder hidden states with the last hidden states from the Encoder
        # outputs => [batch_size, 1, hidden_size]
        # hidden => [num_layers * num_directions, batch_size, hidden_size]
        outputs, hidden = self.gru(embed_con, prev_h)

        # concatenate the embedding vector, output vector from GRU and context vector together
        # to form a tensor of shape [batch_size, 1, hidden_size + hidden_size*2, embed_size]
        output = torch.cat((embed.squeeze(1), outputs.squeeze(1), context.squeeze(1)), dim = 1)

        outputs = outputs.squeeze(1)  # [batch, 1 , hidden_size] => [batch, hidden_size]

        predictions = self.output(outputs)  # [batch, hidden_size] => [batch, trg_vocab_size]
        return predictions, hidden, None

class Seq2Seq(nn.Module):
    """ A Base Seq2Seq Model w/ an Encoder-Decoder Architecture """
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
        hidden = hidden[:self.decoder.num_layers] # [num_layers, batch, 2 * hidden_size]]

        context = hidden[-1].unsqueeze(1) # take the top most layer of the Encoder as the context vector  [batch_size, 1, 2 * hidden_size]

        # first input to the decoder is the <sos> tokens
        input = trg[:, 0]

        for t in range(1, max_seq_len):

            # get translated word for the current timestep
            predictions, hidden, _ = self.decoder(batch=input, prev_h=hidden, context=context, src_mask=src_mask, encoder_hidden=output)
            
            # save to outputs tensor
            outputs[t - 1] = predictions

            pred = predictions.max(1)[1]  # (batch,1)

            # use schedule sampling (with some prob. use teacher forcing or use greedy output from the model)
            input = trg[:, t] if random.random() < tf_ratio else pred

        return outputs.transpose(0, 1) # [batch_size, max_seq_len, trg_vocab_size]
