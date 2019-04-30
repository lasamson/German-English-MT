from torchtext import data, datasets
from torchtext.data import Field, BucketIterator, Iterator
import spacy
import torch
import math

# Why BucketIterator doesn't do us justice?
# Often the sentences aren't of the same length at all, and you
# end up with feeding alot of padding into your network

# Instead, do dynamic batching. If your GPU can process
# 1500 tokens each iteration, and you batch_size is 20, then
# only when you have batches of length 75 will you be
# utilizing all the memory of the GPU.
# An efficient batching mechanism would change the batch size dependiing on the
# sequence length to make sure ~1500 tokens were being processed
# each iteration so you can properly utlize all the memory
# to full capacity

# thus we need to patch the current torch text iterators
# to define a more efficient iterator

# code from http://nlp.seas.harvard.edu/2018/04/03/attention.html
global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """ 
    Keep augmenting batch and calculate the total number of tokens + padding 
    Arguments:
        new: new example to add
        count: current count of examples in the batch
        sofar: current effecctive batch size
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


# batching matters a ton for speed. We want to have very
# evenly divided batches with absolute minimal padding
# This code patches their default batching to make sure we searh over enough sentences
# to bind tight batches
class DataIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shufler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn
                    )
                    for b in random_shufler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(sorted(b, key=self.sort_key)))


def load_dataset(data_path, train_batch_size=4096, dev_batch_size=1, max_len=100):
    """
    This assumes that the data is already pre-processed using Moses Tokenizer
    Returns iterators for the training/dev dataset

    Arguments:
        data_path: path of the dataset
        train_batch_size: batch size of the training data (defined in terms of number of tokens or sentences, depending on the model_type)
        dev_batch_size: batch size of the dev data (usually one)
        max_len: max length of sequeences in a batch
    """

    SRC = Field(tokenize=lambda s: s.split(), init_token="<s>",
                eos_token="</s>", batch_first=True, include_lengths=True)
    TRG = Field(tokenize=lambda s: s.split(), init_token="<s>",
                eos_token="</s>", batch_first=True, include_lengths=True)

    train_data = datasets.TranslationDataset(exts=("train.de", "train.en"), fields=(
        SRC, TRG), path=data_path, filter_pred=lambda x: len(vars(x)['src']) <= max_len and len(vars(x)['trg']) <= max_len)
    dev_data = datasets.TranslationDataset(
        exts=("dev.de", "dev.en"), fields=(SRC, TRG), path=data_path)
    
    SRC.build_vocab(train_data.src, train_data.trg)
    TRG.build_vocab(train_data.src, train_data.trg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use custom DataIterator in order to minimize padding in a sequence
    train_iterator = DataIterator(train_data, batch_size=train_batch_size, device=device,
                                  repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                  batch_size_fn=batch_size_fn, train=True, sort_within_batch=True, shuffle=True)

    # dev_iterator = DataIterator(dev_data, batch_size=dev_batch_size, device=device,
    #                      repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                      batch_size_fn=batch_size_fn, train=False)

    # train_iterator = BucketIterator(train_data, batch_size=train_batch_size, train=True,
    #                                 sort_within_batch=True, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                                 repeat=False, device=device)

    dev_iterator = Iterator(dev_data, batch_size=dev_batch_size,
                            train=False, sort=False, repeat=False, device=device)

    return train_iterator, dev_iterator, SRC, TRG
