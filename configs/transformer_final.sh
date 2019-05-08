# Model Hyperparameters
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
lr=0.0
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
exp_name=$1
model_type=Transformer

# Decoding
average=5
beam_size=5