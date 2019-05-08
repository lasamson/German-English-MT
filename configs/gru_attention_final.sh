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
model_type=GRU

# Decoding
average=10
beam_size=10