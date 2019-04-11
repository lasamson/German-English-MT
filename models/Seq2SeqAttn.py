import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class Encoder(nn.Module):
  def __init__(self, input_size, embed_size, hidden_size, num_layers=1):
    super(Encoder, self).__init__()
    self.input_size = input_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    
    self.embed = nn.Embedding(input_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    
  def forward(self, src, h=None, c=None):
    embedded = self.embed(src)
    if (h and c):
      outputs, (h, c) = self.lstm(embedded, (h, c))
    else:
      outputs, (h, c) = self.lstm(embedded)
    outputs = outputs[:, :, :self.hidden_size] +\
              outputs[:, :, self.hidden_size:]
    return outputs, (h, c)
  

# dot prod attention
class Attention(nn.Module):
  def __init__(self):
    super(Attention, self).__init__()
    
  def forward(self, decoder_hidden, encoder_outputs):
    bsz = encoder_outputs.size(0)
    seq_len = encoder_outputs.size(1)
    d_hid = encoder_outputs.size(2)
    dot_prod = torch.bmm(encoder_outputs[:,0,:].view(bsz, 1, d_hid), decoder_hidden.view(bsz, d_hid, 1)) # initial batch dot prod
    for i in range(1, seq_len): # iterate over seq_len and compute batch dot prod, concat
      new_dot_prod = torch.bmm(encoder_outputs[:,i,:].view(bsz, 1, d_hid), decoder_hidden.view(bsz, d_hid, 1))
      dot_prod = torch.cat([dot_prod, new_dot_prod], 1)
    return dot_prod.transpose(1, 2)
    
  
class Decoder(nn.Module):
  def __init__(self, output_size, embed_size, hidden_size, num_layers=1):
    super(Decoder, self).__init__()
    self.output_size = output_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    
    self.embed = nn.Embedding(output_size, embed_size)
    self.attention = Attention()
    self.lstm = nn.LSTM(hidden_size + embed_size, hidden_size, num_layers, batch_first=True)
    self.output = nn.Linear(hidden_size * 2, output_size)
    
  def forward(self, input, prev_hidden, prev_c, encoder_outputs):
    embedded = self.embed(input).unsqueeze(1) # B x 1 x d_emb
    attn_scores = self.attention(prev_hidden[-1], encoder_outputs) # B x 1 x seq_len
    context = torch.bmm(attn_scores, encoder_outputs) # B x 1 x d_hid
    decoder_input = torch.cat([embedded, context], 2) # B x 1 x d_emb + d_hid
    output, (h, c) = self.lstm(decoder_input, (prev_hidden, prev_c))
    output = output.squeeze(1) # B x 1 x d_hid -> B x d_hid
    context = context.squeeze(1) # B x 1 x d_hid -> B x d_hid
    output = self.output(torch.cat([output, context], 1))
    return output, (h, c), attn_scores


class Seq2SeqAttn(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2SeqAttn, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    
  def forward(self, src, trg, tf_ratio=0.5):
    batch_size = src.size(0) # src, trg shape = B x L
    seq_len = src.size(1)
    vocab_size = self.decoder.output_size
    decoder_outputs = Variable(torch.zeros(batch_size, seq_len, vocab_size))
    
    encoder_output, (h, c) = self.encoder(src)
    h = h[:self.decoder.num_layers]
    c = c[:self.decoder.num_layers]
    decoder_output = Variable(trg.data[:, 0]) # get decoder input for t=0
    for t in range(1, seq_len):
      decoder_output, (h, c), attn_scores = self.decoder(decoder_output,
                                                         h, c, encoder_output)
      print('h', h.size())
      decoder_outputs[:,t] = decoder_output
      teach = random.random() < tf_ratio
      greedy = decoder_output.data.max(1)[1]
      decoder_output = Variable(trg.data[:, t] if teach else greedy)
    return decoder_outputs


# test run
def main():
    encoder = Encoder(10, 2, 3, num_layers=2)
    decoder = Decoder(10, 2, 3, num_layers=1)
    s2s = Seq2SeqAttn(encoder, decoder)
    print(s2s)
    src = torch.randint(0, 9, (4, 5))
    print('Src data', src, src.size())
    trg = torch.randint(0, 9, (4, 7))
    print('Trg data', trg, trg.size())
    output = s2s(src, trg)