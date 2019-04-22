import torch
from torch import nn
from sublayers import LayerNorm, MultiHeadAttention, PositionwiseFeedForwardNet

class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    
    Arguments:
        d_model: hidden size of the encoder
        d_ff: intermediate hidden size of position wise feed forward net
        num_heads: number of self attention heads
        layer_dropout: amount of dropout to apply before layernorm
        attention_dropout: amount of dropout to apply in the multi-head attention layer
        relu_dropout: amount of dropout to apply in positionwise feedforward net
    """
    def __init__(self, d_model, d_ff, num_heads, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.layer_norm_mha = LayerNorm(d_model)
        self.layer_norm_ffn = LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, attention_dropout)
        self.positionwise_feedforward_net = PositionwiseFeedForwardNet(d_model, d_ff, relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
    
    def forward(self, x, src_mask):
        # Layer Normalization before Multi-head attention
        x_norm = self.layer_norm_mha(x)

        # Encoder Multi-head attention
        y = self.multi_head_attention(x_norm, x_norm, x_norm, src_mask)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization before pointwise feedforward net
        x_norm = self.layer_norm_ffn(x)
        
        # Positionwise Feedforward Network
        y = self.positionwise_feedforward_net(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        return y

class DecoderLayer(nn.Module):
    """
    Represents one Decoder Layer of the Transformer Decoder

    Arguments:
        d_model: hidden size of the encoder
        d_ff: intermediate hidden size of position wise feed forward net
        num_heads: number of self attention heads
        layer_dropout: amount of dropout to apply before layernorm
        attention_dropout: amount of dropout to apply in the multi-head attention layer
        relu_dropout: amount of dropout to apply in positionwise feedforward net
    """
    def __init__(self, d_model, d_ff, num_heads, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention_dec = MultiHeadAttention(d_model, num_heads, attention_dropout)
        self.multi_head_attention_enc_dec = MultiHeadAttention(d_model, num_heads, attention_dropout)
        self.positionwise_feedforward_net = PositionwiseFeedForwardNet(d_model, d_ff, relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(d_model)
        self.layer_norm_mha_enc_dec = LayerNorm(d_model)
        self.layer_norm_fnn = LayerNorm(d_model)

    def forward(self, x, encoder_outputs, src_mask, trg_mask):

        # layer normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)

        # Masked Multi-head attention
        y = self.multi_head_attention_dec(x_norm, x_norm, x_norm, trg_mask)

        # Dropout and residual after masked multi-head self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc_dec(x)

        # Multi-head encoder-decoder attention
        y = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, src_mask)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)

        # Layer Normalization before passing to Positionwise Feed Forward Net
        x_norm = self.layer_norm_ffn(x)

        # Dropout and residual after pointwise feedforward network
        y = self.dropout(x + y)

        return y