import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.attention import DotProductAttention, BahdanauAttention
from .transformer.embeddings import Embedder, PositionalEncoder
from .transformer.layers import EncoderLayer, DecoderLayer
from .transformer.sublayers import LayerNorm
from utils.utils import get_clones


class GRUDecoder(nn.Module):
    """ 
    Conditional GRU Decoder that decodes the source sequence into the target sequence
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
        self.pre_output_layer = nn.Linear(embed_size + 2 * hidden_size + hidden_size, hidden_size)

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
        outputs = self.pre_output_layer(outputs)  # [batch, hidden_size] => [batch, hidden_size]
        return outputs, hidden

    def forward(self, trg, encoder_hidden, src_mask, tgt_mask, encoder_final, hidden=None):
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

        outputs = torch.zeros(trg.size(1), trg.size(0), self.hidden_size, device=self.device)

        for t in range(trg.size(1)):
            prev_embed = embed[:, t, :].unsqueeze(1) # [batch, 1, V]
            output, hidden = self.forward_step(prev_embed, encoder_final, hidden, encoder_hidden, src_mask, projected_keys)
            outputs[t] = output
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
    


class TransformerDecoder(nn.Module):
    """
    A Transformer Decoder Module with `num_layers` layers
    Inputs should be in a shape [batch_size, seq_len, hidden_size]
    Outputs will have the shape [batch_size, seq_len, hidden_size]
    Arguments:
        embedding_size: Size of the embeddings
        tgt_vocab_size: Size of the target vocab
        d_model: Hidden size of the Encoder
        num_layers: Total Layers in the Encoder
        num_heads: Number of attention heads
        max_length: Max sequence length
        d_ff: hidden size representation of positionwise feedforward net
        input_dropout: Dropout just after embedding
        layer_dropout: Dropout for each layer
        attention_dropout: Dropout probability after attention 
        relu_dropout: Dropout probability after ReLU operation in FFN
    
    Returns:
        A Tensor of shape [batch_size, seq_len, d_model]
    """
    def __init__(self, embedding_size, tgt_vocab_size, d_model, num_layers, num_heads, max_length, d_ff=2048, input_dropout=0.0, \
                layer_dropout=0.1, attention_dropout=0.1, relu_dropout=0.1):
        super().__init__()

        self.num_layers = num_layers

        # Embeddings and Positional Encodings
        self.embeddings = Embedder(embedding_size, tgt_vocab_size)
        self.positional_encodings = PositionalEncoder(embedding_size, input_dropout=input_dropout)

        # Decoder Stack
        self.decoder_stack = get_clones(DecoderLayer(d_model, d_ff, num_heads, layer_dropout, attention_dropout, relu_dropout), num_layers)

        # Layer Norm on the output of the Decoder
        self.output_layer_norm = LayerNorm(d_model)
    
    def forward(self, trg, encoder_outputs, src_mask, trg_mask, encoder_final=None, hidden=None):
        
        # sum the Embeddings and Positional Encodings
        x = self.positional_encodings(self.embeddings(trg))

        # pass the input through the Decoder Stack
        for i in range(self.num_layers):
            x = self.decoder_stack[i](x, encoder_outputs, src_mask, trg_mask)
        
        # layer norm on the output
        x = self.output_layer_norm(x)

        return x
