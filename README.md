# Neural Machine Translation (German to English)

**Goal**:
To build a neural machine translation model to translate from German to English. Namely, we aim to retrieve coherent English translations from German source sentences using a parallel corpus of German-English sentence pairs as our primary data source. This task is of paramount importance in today’s world, in the fields of academia and industry alike. 

**Motivation**:
To overcome the language barrier and eventually improve communication channels for people worldwide. Translation between languages is of paramount importance in the modern age, in both academic and industrial settings alike.

# Dataset

Bilingual (bitext) data from **IWSLT-2016**, which consists of approximately 200,000 parallel German-English sentence pairs. An example of a German-English sentence pair is illustrated below:	

```
German: Also werde ich über Musikkomposition sprechen, obwohl ich nicht weiß, wo ich anfangen soll.

English: So, I'll talk about musical composition, even though I don't know where to start.
```

**Dataset Statistics:**

| Dataset | Number of sentences |
| ------- | ------------------- |
| Train   | 196884              |
| Dev     | 7883                |

## Preprocessing
Instead of batching by number of sentences, we batch instead by the number of tokens, such that we can most efficiently use the GPU resources (pack each batch as much as possible). We also tokenize the sentences using **Moses Tokenizer**, and encode sentences using **Byte-Pair Encoding** with 32K merge operations, which has a **shared source-target vocabulary** of ~30,000 tokens. 

Inorder to preprocess the **original** IWSLT-16 DE-EN dataset with Moses Tokenizer and apply a shared BPE vocab, run the following script:

```
./scripts/tokenize_and_preprocess_bpe.sh
```

Apply this script, will give you two new folders, namely: **/bpe** and **/tok**

# Attentional GRU Model

The Attentional Encoder-Decoder network  uses a Gated Recurrent Unit (GRU) as the 
Encoder and the Decoder of the Seq2Seq model. We use **Bahdanau** attention to compute context vectors between the decoder hidden state and all encoder hidden states. 

**Encoder**:
2 layer bidirectional GRU with 512 hidden units. 

**Decoder**:
2 layer GRU with 512 hidden units using Bahdanau attentional mechanism 

**Additional Information**:
Since the dataset is relatively small, in order to regularize our model we apply dropout (Variational, Embedding Dropout, Weight Dropout) to various areas of the architecture

**GRU Attention Hyperparamers**:

```
epochs=50
min_freq=1
train_batch_size=4096
dev_batch_size=1
embedding_size=512
hidden_size=512
n_layers_enc=2
n_layers_dec=2
max_len=100
lr=0.001
grad_clip=5.0
tf=1.0
input_dropout=0.1
layer_dropout=0.3
attention=bahdanau
tgt_emb_prj_weight_sharing=True
emb_src_tgt_weight_sharing=True
```

# Transformer Model

We also experiment with a Transformer Encoder-Decoder Architecture that uses a self-attention mechanism to compute representations. We use base model described the **Attention is all you Need** (Vaswani et. al) paper but with slightly modified parameters.

**Encoder & Decoder**: 
6 layer Encoder/Decoder stack with a hidden size of 512 and 8 multi-attention heads.

**Additional Information**:
We apply dropout to various places within the architecture and use label smoothing and same learning rate decay described in the paper.


**Transformer Hyperparameters**:

```
epochs=20
min_freq=1
train_batch_size=4096
dev_batch_size=1
embedding_size=512
hidden_size=512
n_layers_enc=6
n_layers_dec=6
num_heads=8
max_len=100
grad_clip=5.0
tf=1.0
input_dropout=0.1
layer_dropout=0.3
attention_dropout=0.3
relu_dropout=0.3
d_ff=1024
label_smoothing=0.1
n_warmup_steps=16000
tgt_emb_prj_weight_sharing=True
emb_src_tgt_weight_sharing=True
```

# Training & Evaluating
Training and Evaluating models is simply done by making use of the `./scripts/train_eval.sh` script. The script takes in two arguments: first is the **configuration file** (shell script) which should be located in the `./configs/` folder and a **experiment name**. An example of training our transformer with our configurations located at `./configs/transformer_final.sh` and the experiment name `transformer` can be done with this command:

```
./scripts/train_eval.sh ./configs/transformer_final.sh transformer
```

This will create a new folder in the `./experiments/` folder with the name `transformer`. In this folder will be a `params.json` with the configurations for the current experiment, a `train.log` file which contains information related to the training of the model, a `checkpoints/` folder, and a `runs/` folder that will be used by Tensorboard to log metrics (train/val loss and perplexity) related to training.

# Results (BLEU Scores on Dev Set)

| Model                         | Greedy Decoding | Beam Search         |
| ----------------------------- | --------------- | ------------------- |
| Seq2Seq w/ Bahdanau Attention | 31.6            | 33.0 (beam size=10) |
| Transformer                   | 34.0            | 34.5 (beam_size=5)  |
| Boosted GRU                   |                 |                     |



# Things to make note of

- Batching by the number of tokens
- Beam Searh
- GPU (1080Ti)
-
