from torch import nn
import torch


class WeightDrop(nn.Module):
    """
    Apply dropout to the recurrent matrix (weight_hh_l0) of the GRU
    This requires a "hack" in order to work
    Arguments:
        rnn: RNN Network (eg. GRU/LSTM) to apply weight drop (eg. GRU)
        dropout: dropout probability to apply to the GRU
    """

    def __init__(self, rnn, dropout=.4):
        super(WeightDrop, self).__init__()
        self.rnn = rnn
        self.dropout = dropout
        self.weight_name = "weight_hh_l0"

        # this requires a "hack" in order to apply dropout to the recurrent matrix
        # of the GRU
        if issubclass(type(self.rnn), torch.nn.RNNBase):
            self.rnn.flatten_parameters = lambda *args, **kwargs: None

        w = getattr(self.rnn, self.weight_name)
        del self.rnn._parameters[self.weight_name]
        self.rnn.register_parameter(
            self.weight_name + '_original', nn.Parameter(w.data))

    def forward(self, *args):
        # get the original recurrent weight matrix from the lstm module "weight_hh_l0"
        raw_weight = getattr(self.rnn, self.weight_name + "_original")
        # apply dropout to the recurrent weight matrix
        w = nn.Parameter(torch.nn.functional.dropout(
            raw_weight, p=self.dropout, training=self.training))
        # update the weight matrix attribute in the lstm module
        setattr(self.rnn, self.weight_name, w)
        return self.rnn.forward(*args)
