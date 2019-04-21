# Neural Machine Translation (German to English)

# Usage
To train our models:

```
python train_seq2seq.py -data_path="./data/"  -model_dir="./experiments/seq2seq/"
```

To translate German sentences to English
```
python translate.py -data_path="./data/" -model_dir="./experiments/seq2seq/" -model_file="best.pth.tar" -greedy -beam_size=10
```
