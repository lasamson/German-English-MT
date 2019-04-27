import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.encoders import GRUEncoder, TransformerEncoder
from models.decoders import GRUDecoder, TransformerDecoder


class EncoderDecoder(nn.Module):
    """ 
    A Base Seq2Seq Model w/ an Encoder-Decoder Architecture 
    Returns log probabilities of shape [batch_size, seq_len, tgt_vocab_size]
    """

    def __init__(self, encoder, decoder, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, src_lengths=None, trg_lengths=None):

        # pass the src sequence through the Encoder
        encoder_outputs, encoder_final = self.encode(
            src, src_mask, src_lengths)

        if isinstance(self.decoder, GRUDecoder):
            # last hidden state of the Encoder is used as the initial hidden state of the Decoder
            # [num_layers, batch, 2 * hidden_size]]
            encoder_final = encoder_final[:self.decoder.num_layers]
        else:
            encoder_final = None

        # pass the tgt sequence through the Decoder
        decoder_output, _ = self.decode(trg=tgt, encoder_outputs=encoder_outputs,
                                        src_mask=src_mask, trg_mask=tgt_mask,
                                        encoder_final=encoder_final, decoder_hidden=None)
        logits = self.generator(decoder_output)
        return F.log_softmax(logits, dim=-1)

    def encode(self, src, src_mask, src_lengths):
        """ 
        Encode the src sequence using the Encoder 

        Arguments:
            src: Source sequence tensor [batch_size, seq_len] 
            src_mask: Mask for src sequence [batch_size, 1, seq_len]
            src_lengths: lenght of each example in batch

        Returns:
            A Tensor of shape [batch_size, seq_len, hidden_size]
        """
        return self.encoder(src, src_mask, src_lengths)

    def decode(self, trg, encoder_outputs, src_mask, trg_mask, encoder_final=None, decoder_hidden=None):
        """ 
        Decode the target sequence given the outputs from the Encoder         

        Arguments:
            tgt: Target sequence tensor [batch_size, seq_len]
            encoder_outputs: Output Tensor from the Encoder [batch_size, seq_len, hidden_size]
            encoder_final: Final hidden state from the Encoder (only for GRUEncoder)
            src_mask: Mask for src sequence [batch_size, 1, seq_len]
            tgt_mask: Mask for the tgt sequence [batch_size, seq_len, seq_len] (only for TransformerDecoder)
            decoder_hidden: decoder hidden state

        Returns:
            A Tensor of shape [batch_size, seq_len, hidden_size]
        """
        return self.decoder(trg, encoder_outputs, src_mask, trg_mask, encoder_final, hidden=decoder_hidden)


def make_seq2seq_model(params):
    """ Make a Seq2Seq moddel with given parameters """

    if params.model_type == "GRU":
        encoder = GRUEncoder(src_vocab_size=params.src_vocab_size, embed_size=params.embed_size,
                             hidden_size=params.hidden_size, input_dropout_p=params.input_dropout,
                             num_layers=params.n_layers_enc, dropout_p=params.layer_dropout)

        decoder = GRUDecoder(trg_vocab_size=params.tgt_vocab_size, embed_size=params.embed_size,
                             hidden_size=params.hidden_size, attention=params.attention,
                             input_dropout_p=params.input_dropout,
                             dropout_p=params.layer_dropout, device=params.device,
                             num_layers=params.n_layers_dec)
    else:
        encoder = TransformerEncoder(params.embedding_size, params.src_vocab_size, params.hidden_size, params.n_layers_enc,
                                     params.num_heads, params.max_length, d_ff=params.d_ff, input_dropout=params.input_dropout,
                                     layer_dropout=params.layer_dropout, attention_dropout=params.attention_dropout,
                                     relu_dropout=params.relu_dropout)

        # create the Transformer Decoder
        decoder = TransformerDecoder(params.embedding_size, params.tgt_vocab_size, params.hidden_size, params.n_layers_dec,
                                     params.num_heads, params.max_length, d_ff=params.d_ff, input_dropout=params.input_dropout,
                                     layer_dropout=params.layer_dropout, attention_dropout=params.attention_dropout,
                                     relu_dropout=params.relu_dropout)

    # standard linear + softmax generation step
    generator = nn.Linear(params.hidden_size,
                          params.tgt_vocab_size, bias=False)

    if params.tgt_emb_prj_weight_sharing:
        # share the weight matrix between the target embedding and final softmax layer
        if params.model_type == "Transformer":
            generator.weight = decoder.embeddings.embedding.weight
        else:
            generator.weight = decoder.embed.weight

    if params.emb_src_tgt_weight_sharing:
        # Share the weight matrix between source & target word embeddings
        # this can only happen if you have a shared vocab
        assert params.src_vocab_size == params.tgt_vocab_size, "To share word embedding between src and tgt, the vocab sizes must be the same"
        if params.model_type == "Transformer":
            encoder.embeddings.embedding.weight = decoder.embeddings.embedding.weight
        else:
            encoder.embed.weight = decoder.embed.weight

    # define the EncoderDecoer Model
    model = EncoderDecoder(encoder, decoder, generator).to(params.device)

    # weight initialization of the parameters of the model: Xavier Initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
