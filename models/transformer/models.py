""" Full Encoder/Decoder Stack Implementation of the Transformer """
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .embeddings import Embedder, PositionalEncoder
from .layers import EncoderLayer, DecoderLayer
from .sublayers import LayerNorm

def get_clones(module, N):
    """ 
    Produce N identical layers 
    Arguments:
        module: the module (layer) to repeat N times 
        N: number of identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
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

    def forward(self, src, src_mask):

        # sum the Token Embeddings and Positional Encodings
        x = self.positional_encodings(self.embeddings(src))

        # pass the embeddings through the Encoder stack
        for i in range(self.num_layers):
            x = self.encoder_stack[i](x, src_mask)

        # layer norm on the output
        x = self.output_layer_norm(x)

        return x 

class Decoder(nn.Module):
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
                layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super().__init__()

        self.num_layers = num_layers

        # Embeddings and Positional Encodings
        self.embeddings = Embedder(embedding_size, tgt_vocab_size)
        self.positional_encodings = PositionalEncoder(embedding_size, input_dropout=input_dropout)

        # Decoder Stack
        self.decoder_stack = get_clones(DecoderLayer(d_model, d_ff, num_heads, layer_dropout, attention_dropout, relu_dropout), num_layers)

        # Layer Norm on the output of the Decoder
        self.output_layer_norm = LayerNorm(d_model)
    
    def forward(self, trg, encoder_outputs, src_mask, trg_mask):
        
        # sum the Embeddings and Positional Encodings
        x = self.positional_encodings(self.embeddings(trg))

        # pass the input through the Decoder Stack
        for i in range(self.num_layers):
            x = self.decoder_stack[i](x, encoder_outputs, src_mask, trg_mask)
        
        # layer norm on the output
        x = self.output_layer_norm(x)

        return x


class Transformer(nn.Module):
    """
    A standard Encoder-Decoder Architecture for the Transformer Model
    Returns log probabilities of shape [batch_size, seq_len, tgt_vocab_size]

    Arguments:
        encoder: Encoder stack of the Transformer
        decoder: Decoder stack of the Transformer
        generator: linear output softmax layer [d_model, tgt_vocab_size]
    
    Returns:
        A Tensor of shape [batch_size, seq_len, tgt_vocab_size]
    """
    def __init__(self, encoder, decoder, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """ 
        Take in and process masked src and target sequences 

        Arguments: 
            src: Source sequence tensor [batch_size, seq_len]
            tgt: Target sequence tensor [batch_size, seq_len]
            src_mask: Mask for the src sequence [batch_size, 1, seq_len]
            tgt_mask: Mask for the tgt sequence [batch_size, seq_len, seq_len]
        
        Returns:
            A Tensor of shape [batch_size, seq_len, d_model]
        """

        encoder_output = self.encode(src, src_mask) # [batch_size, src_seq_len, d_model]
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask) # [batch_size, tgt_sequence, d_model]
        logits = self.generator(decoder_output) # [batch_size, tgt_sequence, tgt_vocab_size]
        return F.log_softmax(logits, dim=-1)
    
    def encode(self, src, src_mask):
        """
        Encode the source sequence

        Arguments:
            src: Source sequence tensor [batch_size, seq_len]
            src_mask: Mask for src sequence [batch_size, 1, seq_len]

        Returns: 
            A Tensor of shape [batch_size, seq_len, d_model]
        """
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_outputs, src_mask, tgt_mask):
        """
        Decode the target sequence given the outputs from the encoder
        
        Arguments:
            tgt: Target sequence tensor [batch_size, seq_len]
            encoder_outputs: Output Tensor from the Encoder [batch_size, seq_len, d_model]
            src_mask: Mask for src sequence [batch_size, 1, seq_len]
            tgt_mask: Mask for tgt sequence [batch_size, seq_len, seq_len]

        Returns: 
            A Tensor of shape [batch_size, seq_len, d_model]
        """
        return self.decoder(tgt, encoder_outputs, src_mask, tgt_mask)

def make_transformer(params):
    """ Return a Transformer EncoderDecoder Model """

    assert params.embedding_size == params.d_model, "To facilitate the residual connections, \
        the dimensions of all module outputs should be the same. Please make the embedding \
        size and the d_model size the same"

    encoder = Encoder(params.embedding_size, params.src_vocab_size, params.d_model, params.enc_num_layers, 
                    params.num_heads, params.max_length, d_ff=params.d_ff, input_dropout=params.input_dropout,
                    layer_dropout=params.layer_dropout, attention_dropout=params.attention_dropout, 
                    relu_dropout=params.relu_dropout)

    decoder = Decoder(params.embedding_size, params.tgt_vocab_size, params.d_model, params.dec_num_layers, 
                    params.num_heads, params.max_length, d_ff=params.d_ff, input_dropout=params.input_dropout,
                    layer_dropout=params.layer_dropout, attention_dropout=params.attention_dropout, 
                    relu_dropout=params.relu_dropout)

    generator = nn.Linear(params.d_model, params.tgt_vocab_size)

    if params.tgt_emb_prj_weight_sharing:
        # Share the weight matrix between target target word embedding and final logit dense layer
        generator.weight = decoder.embeddings.weight
    
    if params.emb_src_tgt_weight_sharing:
        # Share the weight matrix between source and target word embeddings
        assert params.src_vocab_size == params.tgt_vocab_size, "To share word embedding table, the vocab size of src/tgt should be the same"
        encoder.embeddings.weight = decoder.embeddings.weight

    model = Transformer(encoder, decoder, generator)

    # weight initialization of the parameters of the model: Xavier Initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model