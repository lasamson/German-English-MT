# Neural Machine Translation (German to English)

**Goal**:
To build a neural machine translation model to translate from German to English. Namely, we aim to retrieve coherent English translations from German source sentences using a parallel corpus of German-English sentence pairs as our primary data source. This task is of paramount importance in today’s world, in the fields of academia and industry alike. 

**Motivation**:
To overcome the language barrier and eventually improve communication channels for people worldwide. Translation between languages is of paramount importance in the modern age, in both academic and industrial settings alike.

## Requirements
Before doing anything please install the requirements for this project using the following command: 

```
pip install -r requirements.txt
```

## Getting Started 

In order to reproduce our results, please follow the instructions below:

1. Clone this repo `git clone https://github.com/lasamson/German-English-MT`
2. [Download](https://piazza.com/class_profile/get_resource/jr6ue6jamzn5e7/js50rtixawp4fg) the training and dev data. Make a new `./data/iwslt/` folder in the root directory and place all train and dev files there. 
3. Preprocess and apply BPE to the original dataset: `./scripts/tokenize_and_preprocess_bpe.sh`
4. Train and Evaluate our models by using the following script: `./scripts/train_eval.sh {config_file} {exp_name}`, where **config_file** is the location of a model configuraion file located in `./configs/` and **exp_name** is an experiment name. An example of training our Transformer model is in the [Training & Evaluating](#training-and-evaluating) Section


## Dataset

Bilingual (bitext) data from [**IWSLT-2016**](https://piazza.com/class_profile/get_resource/jr6ue6jamzn5e7/js50rtixawp4fg) DE-EN, which consists of approximately 200,000 parallel German-English sentence pairs. An example of a German-English sentence pair is illustrated below:	

German:
```
Also werde ich über Musikkomposition sprechen, obwohl ich nicht weiß, wo ich anfangen soll.
```

English: 
```
So, I'll talk about musical composition, even though I don't know where to start.
```

**Dataset Statistics:**

| Dataset | Number of sentences |
| ------- | ------------------- |
| Train   | 196884              |
| Dev     | 7883                |
| Test    | 2762                |

Note: The link above for the **IWSLT-16** DE-EN contains only the Train & Dev sets. The Test set can be downloaded with the following [link](https://piazza.com/class_profile/get_resource/jr6ue6jamzn5e7/jv6umkzpzbl647)


### Preprocessing
Instead of batching by number of sentences, we batch instead by the number of tokens, such that we can most efficiently use the GPU resources (pack each batch as much as possible). We also tokenize the sentences using **Moses Tokenizer** (`./utils/tokenizer/tokenizer.perl`), and encode sentences using **Byte-Pair Encoding** with 32K merge operations, which has a **shared source-target vocabulary** of ~30,000 tokens. 

We place all of our original training/dev files in a `./data/iwslt/` folder. These files have to be placed in this manner since our preprocessing script will assume that the data files are located in this specific location. 

Inorder to preprocess the **original** IWSLT-16 DE-EN dataset with **Moses Tokenizer** and apply a **shared BPE vocab**, run the following script:

```
./scripts/tokenize_and_preprocess_bpe.sh
```

An example of an original sequence and a sequence applied with BPE is given below:

**Original Sequence**: 
```
David Gallo : This is Bill Lange . I 'm Dave Gallo .
```

**Sequence applied with BPE**:
```
David Gall@@ o : This is Bill Lange . I '@@ m Da@@ ve Gall@@ o .
```

Applying the `tokenize_and_preprocess_bpe.sh` script will give create two new folders, namely: `/bpe` and `/tok` in the `./data/iwslt`
folder
```
./data
│
└───iwslt
│   │   dev.de
│   │   dev.en 
│   │   train.de
│   │   train.en 
│   │
│   └───bpe
│       │   dev.de 
│       │   dev.en 
│       │   ...
│   └───tok 
│       │   dev.de.atok  
│       │   dev.en.atok  
│       │   ...  
│   
```

## Models
### Attentional GRU Model

The Attentional GRU uses a Gated Recurrent Unit (GRU) as the 
Encoder and the Decoder of the Seq2Seq model. We use **Bahdanau** attention to compute context vectors between the decoder hidden state and all encoder hidden states. 

**Encoder**:
2 layer bidirectional GRU with 512 hidden units. 

**Decoder**:
2 layer GRU with 512 hidden units using Bahdanau attentional mechanism 

**Additional Information**:
Since the dataset is relatively small, in order to regularize our model we apply dropout (Variational, Embedding Dropout, Weight Dropout) to various areas of the architecture. Since we are using a shared vocab, we can also tie the weights for both the Encoder/Decoder Embedding layers and also the softmax layer. This also regularizes the model by reducing the number of parameters.

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

### Transformer Model

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

### Boosted GRU
Lastly, we also experimented with **Boosting** (Zhang et. al) our dataset, by duplicating 10% of the hardest examples in the dataset for each epoch. The intuition behind this idea is that for any problem, some data points are harder to learn than others. However, during training, models treat each data point equally. Thus, it makes sense to make the model spend more time on harder training examples instead, and this is achieved by duplicating hard examples.

**Encoder**:
2 layer bidirectional GRU with 512 hidden units. 

**Decoder**:
2 layer GRU with 512 hidden units using Bahdanau attentional mechanism 

**Additional Information**:
The hardness of a data point is calculated using the average perplexity of the sentence. Intuitively, it makes sense that an example with high perplexity is difficult for the model to classify, and thus the model should spend more time on it. 

**Transformer Hyperparameters**:
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
boost_percent=0.1
boost_warmpup=0
exp_name=$1
model_type=GRU
```



## Training and Evaluating
Training and Evaluating models is simply done by making use of the `./scripts/train_eval.sh` script. The script takes in two arguments: first is the **configuration file** (shell script) which should be located in the `./configs/` folder and a **experiment name**. An example of training & evaluating our transformer with our configurations located at `./configs/transformer_final.sh` and the experiment name `transformer` can be done with this command:

```
./scripts/train_eval.sh ./configs/transformer_final.sh transformer
```

This will create a new folder in the `./experiments/` folder with the name `transformer`. In this folder will be a `params.json` with the configurations for the current experiment, a `train.log` file which contains information related to the training of the model, a `checkpoints/` folder, a `outputs/` folder that will contain translations of the dev set using either **Beam Search** or **Greedy Decoding**, and a `runs/` folder that will be used by Tensorboard to log metrics (train/val loss and perplexity) related to training.

```
./experiments
│
└───transformer
│   │   train.log
│   │   params.json 
│   │
│   └───checkpoints
│       │   epoch_1.pth.tar 
│       │   epoch_2.pth.tar
│       │   ...
│   └───runs
│       │   events.out.tfevents.1557291704.node057.8801.0   
│   └───outputs 
│       │   beam_search_outputs_size=10.en.final  
│       │   beam_search_outputs_size=5.en.final  
│       │   greedy_outputs.en.final  
│   
└───gru_attention
│   │   ...
│
```

**Note**: if you already have a trained model in a `./experiments/exp_name/checkpoints` folder, you can you evaluate a model by simply calling the `translate.py` script:

```
python evaluate.py -data_path="./data/iwslt/bpe/" 
                   -model_dir="./exeriments/{exp_name}/" 
                   -model_file="{model_file}" 
                   -beam_size={beam_size}
```

This script assumes that the `model_file` is in the `./experiments/exp_name/checkpoints` folder. If you however want to average the last **n** checkpoints for evaluation, 
you can forgo the `-model_file` argument and use the `-average` argument instead. This is done in the following manner:

```
python evaluate.py -data_path="./data/iwslt/bpe/" 
                   -model_dir="./exeriments/{exp_name}/" 
                   -average={n} 
                   -beam_size={beam_size}
```

### Description of Hyperparameters

| Hyperparameter             | Description                                                                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| train_batch_size           | The batch size of the training data (# of tokens)                                                                                                     |
| dev_batch_size             | The batch size of the dev data (# of tokens)                                                                                                          |
| embedding_size             | Embedding size                                                                                                                                        |
| n_layers_enc               | Number of layers for the Encoder                                                                                                                      |
| n_layers_dec               | Number of layers for the Decoder                                                                                                                      |
| max_len                    | Maximum sequence length                                                                                                                               |
| lr                         | Learning Rate                                                                                                                                         |
| grad_clip                  | The max norm of the gradients                                                                                                                         |
| tf                         | The probability of using teacher-forcing during training                                                                                              |
| input_dropout              | Dropout applied to the Embeddings (for Attentional GRU, this applies Embedding Dropout)                                                               |
| layer_dropout              | For Attentional GRU, this is the dropout applied inbetween layers (Variational Dropout). For Transformer, this is the dropout applied after LayerNorm |
| relu_dropout               | Dropout applied after the ReLU non-linearity in the Positionwise Feedforward Net (only applies to the Transformer)                                    |
| attention_dropout          | Dropout applied to the attention scores matrix (only applies to the Transformer)                                                                      |
| num_heads                  | Number of heads in the multi-head atention (only applies to the Transformer)                                                                          |
| d_ff                       | Hidden dimensionality of the Positionwise Feedforward Net (only applies to the Transformer)                                                           |
| label_smoothing            | Label smoothing parameter (only applies to the Transformer)                                                                                           |
| n_warmup_steps             | Number of warmup steps in the learning rate decay scheme used in the "Attention is all you need" paper (only applies to the Transformer)              |
| attention                  | Type of attention to use for Attentional GRU models (eg. Bahdanau or Dot)                                                                             |
| tgt_emb_prj_weight_sharing | Whether to tie the weights of the decoder embedding layer and the softmax linear layer                                                                |
| emb_src_tgt_weight_sharing | Whether to tie the weights of the encoding embedding layer and the decoder embedding layer                                                            |
| boost_percent              | Percentage of hardest examples to duplicate                                                                                                           |
| boost_warmup               | How many epochs to go without boosting before starting to boost                                                                                       |
| boost                      | Whether or not to boost the model                                                                                                                     |
| beam_size                  | Size of the beam search                                                                                                                               |
| average                    | Number of checkpoints to average for evaluation                                                                                                       |


### Hardware
All of our models were trained on a single 1080Ti GPU.

### Using Tensorboard
The training process for all experiments can be visualized with Tensorboard. In order to run Tensorboard to visualize all experiments, run `tensorboard -logdir=experiments/` in the root directory. This will create a new Tensorboard server and can be visualized using **localhost** with the port (default is port 6006) specified by Tensorboard.

## Results

### BLEU scores on the Dev Set

| Model                         | Greedy Decoding | Beam Search         |
| ----------------------------- | --------------- | ------------------- |
| Seq2Seq w/ Bahdanau Attention | 31.6            | 33.0 (beam size=10) |
| Transformer                   | 34.0            | 34.5 (beam_size=5)  |
| Boosted GRU                   |                 |                     |

## Things to make note of

- Batching by the number of tokens
- Beam Search
- GPU (1080Ti)
-
