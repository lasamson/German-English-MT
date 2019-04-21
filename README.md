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

# Results (BLEU Scores on Dev Set)

| Model                            	| Greedy Decoding 	| Beam Search  	|
|----------------------------------	|-----------------	|--------------	|
| Seq2Seq                          	|                 	|              	|
| Seq2Seq w/ Bahdanau Attention    	|                 	|              	|
| Seq2Seq w/ Dot Product Attention 	|                 	|              	|
| Transformer 	                    |                 	|              	|
| Transformer with LISA             |                   |               |
