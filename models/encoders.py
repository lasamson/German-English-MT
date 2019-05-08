import random
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .transformer.embeddings import Embedder, PositionalEncoder
from .transformer.layers import EncoderLayer, DecoderLayer
from .transformer.sublayers import LayerNorm
from utils.utils import get_clones
from .dropout.embed_dropout import embedded_dropout
from .dropout.weight_drop import WeightDrop
from .dropout.variational_dropout import VariationalDropout


class GRUEncoder(nn.Module):
    """ 
    GRU Encoder that encodes the source sentence into a fixed sized representation 

    Arguments:
        src_vocab_size: Size of the SRC vocab
        embed_size: Embedding Size for the Encoder
        hidden_size: Hidden size of the Encoder
        input_dropout_p: Amount of dropout applied to the embedding layer (dropout full words)
        dropout_p: Dropout applied inbetween layers of the Encoder, using Variational Dropout
        num_layers: Number of layers of the Encoder
    """

    def __init__(self, src_vocab_size, embed_size, hidden_size, input_dropout_p, dropout_p, num_layers=1):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p
        self.embed = nn.Embedding(src_vocab_size, embed_size)
        self.variational_dropout = VariationalDropout()

        # need to split the Encoder into seperate layers inroder to apply
        # Weight Dropout and Variational Dropout
        # Note: since this Encoder is bidirectional, the input to the layers > 1
        # are going to be 2 x hidden_size, since the directions are concatenated and not summed
        self.gru_list = [WeightDrop(nn.GRU(input_size=self.embed_size if num_layer == 0 else 2*self.hidden_size,
                                           hidden_size=hidden_size, bidirectional=True, batch_first=True, num_layers=1))
                         for num_layer in range(self.num_layers)]
        self.gru = nn.ModuleList(self.gru_list)

    def forward(self, src, src_mask, src_lengths=None):
        # Apply Embedding Dropout to dropout full words with some probability
        # Embed ==> [batch, seq_len, V]
        embed = embedded_dropout(self.embed, src,
                                 dropout=self.input_dropout_p if self.training else 0)

        # Packs a Tensor containing padded sequences of variable length.
        # make sure the RNN doesn't perform computation the padded elements
        # not wasting comoputing
        encoder_outputs = embed
        final_hidden_states = []
        for l, gru in enumerate(self.gru):

            # pack the padded the sequence
            encoder_outputs = pack_padded_sequence(
                encoder_outputs, src_lengths, batch_first=True)

            # pass through GRU
            encoder_outputs, encoder_final = gru(encoder_outputs)
            final_hidden_states.append(encoder_final)

            # undo packing to get encoder_outputs => [batch_size, seq_len, hidden_size]
            encoder_outputs, _ = pad_packed_sequence(
                encoder_outputs, batch_first=True)

            if l != self.num_layers - 1:
                # apply variational dropout to the output for the RNN
                encoder_outputs = self.variational_dropout(encoder_outputs,
                                                           dropout=self.dropout_p)

        # concatenate the final hidden states from all layres in the Encoder
        encoder_final = torch.cat(final_hidden_states, dim=0)
        encoder_final = encoder_final.view(
            self.num_layers, 2, src.size(0), self.hidden_size).transpose(1, 2).contiguous().view(self.num_layers, src.size(0), -1)

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

    def __init__(self, embedding_size, src_vocab_size, d_model, num_layers, num_heads, max_length, d_ff=2048, input_dropout=0.0,
                 layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super().__init__()

        self.num_layers = num_layers

        # Embeddings and Postional Encodings
        self.embeddings = Embedder(
            d_model=embedding_size, vocab_size=src_vocab_size)
        self.positional_encodings = PositionalEncoder(
            embedding_size, input_dropout=input_dropout)

        # Make the Encoder Stack with `num_layers` layers
        self.encoder_stack = get_clones(EncoderLayer(
            d_model, d_ff, num_heads, layer_dropout, attention_dropout, relu_dropout), num_layers)

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
