""" Implementation of Embedding Dropout """
from torch import nn
from torch.autograd import Variable
import torch


def embedded_dropout(embed, batch, dropout=.1):
    """ 
    Apply Embedding Dropout (dropping entire words) to the 
    Embedding Matrix and return a new tensor where the 
    new dropped embedding matrix is applied on the batch

    Arguments:
        - embed: torch.nn.Embedding matrix
        - batch: a batch of inputs of size (N, L) where N is 
        the number of examples in the batch and L is the max sequence
        lenght
        - dropout: amount of dropout to apply to the embedding matrix
    Returns:
        - pytorch tensor of size (N, L, D) where D is the embedding dimension.
    """

    V, D = embed.weight.size()

    embedding_weight = embed.weight

    # create a binary mask of size V, since will be expanding the whole word from the
    # embedding matrix
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create a binary mask of size [V, 1]
    # essentially this mask says which words from the vocab will
    # be dropped from the whole batch
    keep_prob = 1 - dropout
    mask = torch.FloatTensor(V, 1).to(device).bernoulli_(keep_prob)

    # expand the binary mask into the size of the embeddig matrix [V, D]
    mask = mask.expand(V, D)

    mask = mask / keep_prob  # use inverted dropout at train time

    # apply the mask to the embedding weight
    # dropping out random words from the weight matrix
    new_embedding_weight = embedding_weight * mask

    return nn.functional.embedding(batch, new_embedding_weight, -1, embed.max_norm,
                                   embed.norm_type, embed.scale_grad_by_freq, embed.sparse)
