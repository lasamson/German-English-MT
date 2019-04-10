from torchtext import data, datasets
from torchtext.data import Field, BucketIterator
import spacy
import torch

def load_dataset(data_path, min_freq=2, batch_size=128):
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

    SRC = Field(tokenize=tokenize_de, init_token="<BOS>", eos_token="<EOS>")
    TRG = Field(tokenize=tokenize_en, init_token="<BOS>", eos_token="<EOS>")

    train_data, dev_data = datasets.TranslationDataset.splits(exts=(".de", ".en"),
                                    fields=(SRC, TRG), path=data_path, test=None, validation="dev")

    SRC.build_vocab(train_data, min_freq=min_freq)
    TRG.build_vocab(train_data, min_freq=min_freq)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, dev_iterator = BucketIterator.splits((train_data, dev_data),
                                                         batch_size=batch_size,
                                                         device=device)
    return train_iterator, dev_iterator, SRC, TRG
