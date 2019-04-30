from torch import nn
from torch.nn.utils.rnn import PackedSequence
import torch


class VariationalDropout(nn.Module):
    """ 
    Apply Variational Dropout to a Recurrent Neural Network 
    We need to make sure we create the same binary mask across all
    timesteps in a single layer. This mask will only apply dropout to a single 
    layer so we need to split all layers in the RNN implemention
    for this to work.
    """

    def __init__(self):
        super(VariationalDropout, self).__init__()

    def forward(self, x, dropout=.5):

        # if you aren't in training mode, then we don't want to use dropout
        assert 0 <= dropout <= 1, "Dropout must be in between 0 and 1"

        if not self.training:
            return x

        bsz, seq_len, input_dim = x.size()

        # sample of binary mask of size [bsz, 1, input_dim]
        # this ensures that we sample a different binary mask
        # for each example in the batch and each binary mask on the
        # input is same through all timestep
        keep_prob = 1 - dropout
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = torch.FloatTensor(
            bsz, 1, input_dim).to(device).bernoulli_(keep_prob)

        mask = mask / keep_prob  # use inverted dropout

        # expand the mask such that the binary mask
        # is the same through all **timesteps** in the sequence
        # apply the mask to the input
        mask = mask.expand(bsz, seq_len, input_dim)
        x = mask * x
        return x
