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

    def forward(self, batch, src_mask, src_lengths=None):
        embed = self.dropout(self.embed(batch))
        packed = pack_padded_sequence(embed, src_lengths, batch_first=True)
        
        # output => [batch_size, seq_len, num_directions * hidden_size]
        # hidden => [num_layers * num_directions, batch_size, hidden_size]
        encoder_outputs, encoder_final = self.gru(packed) 
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)

        # we need to manually concatenate the final states for both directions
        # the outputs are from the last layer in the stacked LSTM
        fwd_hidden = encoder_final[0:encoder_final.size(0):2]
        bwd_hidden = encoder_final[1:encoder_final.size(0):2]
        encoder_final = torch.cat([fwd_hidden, bwd_hidden], dim=2)  # [num_layers, batch, 2*hidden_size]
        return encoder_outputs, encoder_final


class Decoder(nn.Module):
    """ 
    Conditional GRU Decoder 
    """
    def __init__(self, trg_vocab_size, embed_size, hidden_size, attention, input_dropout_p, dropout_p, device, bridge=True, num_layers=1):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.embed = nn.Embedding(trg_vocab_size, embed_size)
        self.attention = DotProductAttention(hidden_size=hidden_size) if attention == "dot" else BahdanauAttention(hidden_size=hidden_size) if attention is not None else None
        self.dropout = nn.Dropout(input_dropout_p)
        self.gru = nn.GRU(embed_size + 2 * hidden_size, hidden_size, num_layers, dropout=dropout_p, batch_first=True)
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None
        self.output = nn.Linear(embed_size + 2 * hidden_size + hidden_size, trg_vocab_size)

    def forward_step(self, prev_embed, encoder_final, hidden, encoder_hidden=None, src_mask=None, projected_keys=None):
        """
        Perform a single step of decoding
        
        Arguments:
            prev_embed: embedding input at current timestep of decoding [batch_size, 1, V]
            encoder_final: final hidden state of the Encoder [num_layers, batch_size, 2*hidden_size]
            hidden: hidden state of the GRU [num_layers * num_directions, batch_size, hidden_size]
            encoder_hidden: hidden states of the Encoder [batch_size, seq_len, 2 * hidden_size]
            src_mask: Mask for src sequence [batch_size, 1, seq_len]
            projected_keys: The projected Encoder hidden states [batch_size, seq_len, 2 * hidden_size]
        """

        if self.attention is not None:
            assert src_mask is not None, "Src mask must be passed in, if you are using attention"
            
            # use the top most layer of the Decoder as the query
            # query => [batch_size, 1, hidden_size]
            query = hidden[-1].unsqueeze(1)

            # context: [batch_size, 1, 2*hidden_size], attention_scores: [batch_size, 1, 2*hidden_size]
            context, _ = self.attention(
                query=query, projected_keys=projected_keys,
                values=encoder_hidden, mask=src_mask)
        else:
            context = encoder_final[-1].unsqueeze(1) # [batch_size, 1, 2*hidden_size]

        # update the GRU input
        embed_con = torch.cat((prev_embed, context), dim=2)

        # run the input through the GRU for one timestep
        outputs, hidden = self.gru(embed_con, hidden)

        # concatenate the embedding vector, output vector from GRU and context vector together
        # to form a tensor of shape [batch_size, 1, hidden_size + hidden_size*2, embed_size]
        outputs = torch.cat((prev_embed.squeeze(1), outputs.squeeze(1), context.squeeze(1)), dim = 1)
        outputs = outputs.squeeze(1)  # [batch, 1 , hidden_size] => [batch, hidden_size]
        predictions = self.output(outputs)  # [batch, hidden_size] => [batch, trg_vocab_size]

        return predictions, hidden

    def forward(self, trg, encoder_hidden, encoder_final, hidden=None, src_mask=None):
        embed = self.dropout(self.embed(trg))  # [batch, seq_len, V]

        # initialize the decoder hidden state using the final encoder hidden state
        # [num_layers, batch_size, 2*hidden_size] => [num_layers, batch_size, hidden_size]
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # project the keys (encoder hidden states)
        # projected_keys => [batch_size, seq_len, 2*hidden_size]

        if self.attention is not None:
            projected_keys = self.attention.key_layer(encoder_hidden)
        else:
            projected_keys = None

        outputs = torch.zeros(trg.size(1), trg.size(0), self.trg_vocab_size, device=self.device)

        for t in range(trg.size(1)):
            prev_embed = embed[:, t, :].unsqueeze(1) # [batch, 1, V]
            predictions, hidden = self.forward_step(prev_embed, encoder_final, hidden, encoder_hidden, src_mask, projected_keys)
            outputs[t] = predictions

        return outputs.transpose(0, 1)
    
    def init_hidden(self, encoder_final):
        """
        Returns the initial decoder state,
        conditioned on the final encoder state.
        Since the hidden size of the encoder 
        can be different from the size of the decoder
        we need a linear transformation from the 
        Encoder's hidden size to the Decoder's hidden
        size
        """
        if encoder_final is None:
            return None
        return torch.tanh(self.bridge(encoder_final)) # [num_layers, batch_size, 2*hidden_size] => [num_layers, batch_size, hidden_size]
    


class Seq2Seq(nn.Module):
    """ A Base Seq2Seq Model w/ an Encoder-Decoder Architecture """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        encoder_outputs, encoder_final = self.encode(src, src_mask, src_lengths)

        # last hidden state of the Encoder is used as the initial hidden state of the Decoder
        encoder_final = encoder_final[:self.decoder.num_layers] # [num_layers, batch, 2 * hidden_size]]

        return self.decode(trg, encoder_outputs, encoder_final, src_mask)

    def encode(self, src, src_mask, src_lengths):
        """ Encode the src sequence using the Encoder """
        return self.encoder(src, src_mask, src_lengths)
    
    def decode(self, trg, encoder_outputs, encoder_final, src_mask, decoder_hidden=None):
        """ Decode the trg sequence using the Decoder """
        return self.decoder(trg, encoder_outputs, encoder_final, hidden=decoder_hidden, src_mask=src_mask)