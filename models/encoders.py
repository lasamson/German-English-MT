import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .transformer.embeddings import Embedder, PositionalEncoder
from .transformer.layers import EncoderLayer, DecoderLayer
from .transformer.sublayers import LayerNorm
from utils.utils import get_clones

class GRUEncoder(nn.Module):
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

    def forward(self, src, src_mask, src_lengths=None):
        embed = self.dropout(self.embed(src))
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


class TransformerEncoder(nn.Module):
    """
    A Transformer Encoder Module with `num_layers` layers
    Inputs to the Encoder should in the shape [batch_size, seq_len, hidden_size]
    Outputs of the Encoder will have the shape [batch_size, seq_len, hidden_size]

    Arguments:
        embedding_size: Size of the embeddings
        src_vocab_size: Size of the source vocab
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
    def __init__(self, embedding_size, src_vocab_size, d_model, num_layers, num_heads, max_length, d_ff=2048, input_dropout=0.0, \
                layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super().__init__()

        self.num_layers = num_layers

        # Embeddings and Postional Encodings
        self.embeddings = Embedder(d_model=embedding_size, vocab_size=src_vocab_size)
        self.positional_encodings = PositionalEncoder(embedding_size, input_dropout=input_dropout)
    
        # Make the Encoder Stack with `num_layers` layers
        self.encoder_stack = get_clones(EncoderLayer(d_model, d_ff, num_heads, layer_dropout, attention_dropout, relu_dropout), num_layers)

        # Layer Norm on the output of the Encoder
        self.output_layer_norm = LayerNorm(d_model)

    def forward(self, src, src_mask, src_lengths=None):

        # sum the Token Embeddings and Positional Encodings
        x = self.positional_encodings(self.embeddings(src))

        # pass the embeddings through the Encoder stack
        for i in range(self.num_layers):
            x = self.encoder_stack[i](x, src_mask)

        # layer norm on the output
        x = self.output_layer_norm(x)

        return x, None
