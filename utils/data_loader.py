from torchtext import data, datasets
from torchtext.data import Field, BucketIterator, Iterator
import spacy
import torch


def load_dataset(data_path, min_freq=5, train_batch_size=32, dev_batch_size=1):
    """
    Returns iterators for the training/dev dataset

    Arguments:
        data_path: path of the dataset
        min_freq: min freq. needed to include a token in the vocabulary
        batch_size: size of the batch for BucketIterator
    """

    spacy_de = spacy.load("de")
    spacy_en = spacy.load("en")

    def tokenize_de(text):
        """ Tokenize German text """
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        """ Tokenize English text """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize=tokenize_de, init_token=None, eos_token="</s>", batch_first=True, include_lengths=True)
    TRG = Field(tokenize=tokenize_en, init_token="<s>", eos_token="</s>", batch_first=True, include_lengths=True)

    MAX_LEN = 50
    train_data = datasets.TranslationDataset(exts=("train.de", "train.en"), fields=(SRC, TRG), path=data_path, filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
    dev_data = datasets.TranslationDataset(exts=("dev.de", "dev.en"), fields=(SRC, TRG), path=data_path)

    # train_data, dev_data = datasets.TranslationDataset.splits(exts=(".de", ".en"),
    #                                 fields=(SRC, TRG), path=data_path, test=None, validation="dev",
    #                                 filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)

    SRC.build_vocab(train_data.src, min_freq=min_freq)
    TRG.build_vocab(train_data.trg, min_freq=min_freq)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator = BucketIterator(train_data, batch_size=train_batch_size, train=True,
                                    sort_within_batch=True, sort_key=lambda x: (len(x.src), len(x.trg)),
                                    repeat=False, device=device)

    dev_iterator = Iterator(dev_data, batch_size=dev_batch_size, train=False, sort=False, repeat=False, device=device)
    return train_iterator, dev_iterator, SRC, TRG
