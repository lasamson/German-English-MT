# Setup some hyperparametess for the Baseline Transformer Model 

# Data Folders
data_path=./data/iwslt/bpe/

# Model Hyperparams
epochs=20 
min_freq=1
train_batch_size=4096
dev_batch_size=1
embedding_size=256
hidden_size=256
n_layers_enc=2
n_layers_dec=2
num_heads=8
max_len=100
lr=0.0
grad_clip=5.0
tf=1.0
input_dropout=0.1
layer_dropout=0.2
attention_dropout=0.2
relu_dropout=0.2
d_ff=1024
label_smoothing=0.1
n_warmup_steps=1000
tgt_emb_prj_weight_sharing=True
emb_src_tgt_weight_sharing=True
exp_name=$1
model_type=Transformer

echo $1

model_dir="./experiments/"$exp_name"/"

# run make_config.py to make the model folder
echo "Make experiment folder for exp_name: " $exp_name
python ./make_config.py -e $epochs -mf $min_freq -trbsz $train_batch_size -dvbsz $dev_batch_size \
                        -embsz $embedding_size -hidsz $hidden_size -nenc $n_layers_enc -ndec $n_layers_dec \
                        -maxlen $max_len -lr $lr -gc $grad_clip -tf $tf -inpdrop $input_dropout -laydrop $layer_dropout  \
                        -attndrop $attention_dropout -reldrop $relu_dropout -labsmooth $label_smoothing -dff $d_ff \
                        -decweightshare $tgt_emb_prj_weight_sharing -encdecweightshare $emb_src_tgt_weight_sharing \
                        -warmup $n_warmup_steps -exp_name $exp_name -model_type $model_type
echo "Done making experiment folder..."

echo "Training experiment: " $exp_name " using model_type: " $model_type
python train.py -data_path=$data_path -model_dir=$model_dir