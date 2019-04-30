# Setup some hyperparametess for the GRU Seq2Seq w/ Attention Baseline Model

# Data Folders
data_path=./data/iwslt/bpe/

# Model Hyperparameters
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
exp_name=$1
model_type=GRU

# Decoding
average=10
beam_size=10

model_dir="./experiments/"$exp_name"/"

# run make_config.py to make the model folder
echo "Make experiment folder for exp_name: " $exp_name
python ./make_config.py -e $epochs -mf $min_freq -trbsz $train_batch_size -dvbsz $dev_batch_size \
                        -embsz $embedding_size -hidsz $hidden_size -nenc $n_layers_enc -ndec $n_layers_dec \
                        -maxlen $max_len -lr $lr -gc $grad_clip -tf $tf -inpdrop $input_dropout -laydrop $layer_dropout  \
                        -decweightshare $tgt_emb_prj_weight_sharing -encdecweightshare $emb_src_tgt_weight_sharing \
                        -attention $attention -exp_name $exp_name -model_type $model_type
echo "Done making experiment folder..."

echo "Training experiment: " $exp_name " using model_type: " $model_type
python train.py -data_path=$data_path -model_dir=$model_dir

echo "Evaluating Model on Dev set using Greedy/Beam Search..."
python translate.py -data_path=$data_path -model_dir=$model_dir -average=$average -greedy -beam_size=$beam_size
