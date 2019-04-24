""" Full Encoder/Decoder Stack Implementation of the Transformer """
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_clones
from .embeddings import Embedder, PositionalEncoder
from .layers import EncoderLayer, DecoderLayer
from .sublayers import LayerNorm


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
    
    def forward(self, src, trg, src_mask, trg_mask):
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
        decoder_output = self.decode(trg, encoder_output, src_mask, trg_mask) # [batch_size, tgt_sequence, d_model]
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
    
    # create the Transformer Encoder
    encoder = Encoder(params.embedding_size, params.src_vocab_size, params.d_model, params.enc_num_layers, 
                    params.num_heads, params.max_length, d_ff=params.d_ff, input_dropout=params.input_dropout,
                    layer_dropout=params.layer_dropout, attention_dropout=params.attention_dropout, 
                    relu_dropout=params.relu_dropout)

    # create the Transformer Decoder
    decoder = Decoder(params.embedding_size, params.tgt_vocab_size, params.d_model, params.dec_num_layers, 
                    params.num_heads, params.max_length, d_ff=params.d_ff, input_dropout=params.input_dropout,
                    layer_dropout=params.layer_dropout, attention_dropout=params.attention_dropout, 
                    relu_dropout=params.relu_dropout)

    # create the generator (linear softmax layer)
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