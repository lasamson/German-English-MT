#!/bin/bash
#
# Preprocess train and validation data with BPE

###########################################
# Some hyperparameters
workdir="."
datadir=$workdir/data/iwslt
tokdir=$datadir/tok

# make the directory with tokenized files
mkdir -p $tokdir 

src="de"
trg="en"

train_tok="train"
valid_tok="dev"

# save the the new bpe files in this folder
mkdir -p $datadir/bpe


# output files from applying the BPE vocab's 
train_bpe_src=$datadir/bpe/train.$src
valid_bpe_src=$datadir/bpe/dev.$src
train_bpe_trg=$datadir/bpe/train.$trg
valid_bpe_trg=$datadir/bpe/dev.$trg

# BPE Vocab
bpe_vocab=$datadir/bpe/bpe.32000

###########################################
echo `date '+%Y-%m-%d %H:%M:%S'` "- Tokenizing text using Moses Tokenizer"
# (1) Tokenize text using Moses Tokenizer
for l in en de; do for f in $datadir/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $tokdir/"$(cut -d'/' -f4 <<<$f)".atok; done; done

###########################################

# (2) Apply the BPE to the Train/Dev Data
echo `date '+%Y-%m-%d %H:%M:%S'` "Apply BPE to training data" 
subword-nmt apply-bpe -c $bpe_vocab --input $tokdir/${train_tok}.$src.atok --output $train_bpe_src 
subword-nmt apply-bpe -c $bpe_vocab --input $tokdir/${train_tok}.$trg.atok --output $train_bpe_trg  

echo `date '+%Y-%m-%d %H:%M:%S'` "Apply BPE to dev data" 
subword-nmt apply-bpe -c $bpe_vocab --input $tokdir/${valid_tok}.$src.atok --output $valid_bpe_src 
subword-nmt apply-bpe -c $bpe_vocab --input $tokdir/${valid_tok}.$trg.atok --output $valid_bpe_trg  

###########################################

echo `date '+%Y-%m-%d %H:%M:%S'` "- Done with preprocess_bpe.sh"