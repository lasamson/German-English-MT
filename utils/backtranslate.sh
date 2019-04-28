#!/bin/bash

# arguments
# src2trg_model_path=$1
num_synth=$1
# src2trg_data=$3

# directories
workdir="."
datapath=$workdir/data
pretrained_model=$datapath/wmt18ensemble/wmt18.model1.pt
monolingual=$datapath/europarl-v7.de-en
bpepath=$datapath/iwslt/bpe
binpath=$datapath/iwslt/bin

echo $datapath
echo $pretrained_model
echo $monolingual
echo $bpepath
echo $binpath

mkdir -p $binpath

# filter for subset of monolingual data
head -$num_synth $monolingual/europarl-v7.de-en.en > $monolingual/monolingual.en
head -$num_synth $monolingual/europarl-v7.de-en.de > $monolingual/monolingual.de
echo "monolingual data filtered..."

# preprocess monolingual data
echo "preprocessing monolingual data..."
fairseq-preprocess --source-lang en --target-lang de --only-source \
    --trainpref $monolingual --destdir $monolingual
echo "monolingual data preprocessed..."

echo "generating translations..."
fairseq-generate $monolingual --path $pretrained_model --batch-size 128 --beam 1 --target-lang de
echo "translations generated..."


