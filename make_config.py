import argparse
import os

def write_params(args, dir_name, model2args):
    f = open(dir_name + '/params.json', 'w+',)

    model_type = getattr(args, 'model_type')

    f.write('{\n')
    for idx, arg in enumerate(vars(args)):
        if arg in model2args[model_type]:
            if isinstance(getattr(args, arg), str):
                f.write('\t"{}": "{}"'.format(arg, getattr(args, arg)))
            else:
                f.write('\t"{}": {}'.format(arg, getattr(args, arg)))
            if idx != len(vars(args))-1:
                f.write(",\n")
            print(arg, getattr(args, arg))

    f.write('\n}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Set up directory for experiments')
    p.add_argument('-e', '--epochs', type=int, help='Number of epochs to train', required=True)
    p.add_argument('-mf', '--min_freq', type=int, default=5, help='Minimum frequency cutoff for word to be included in vocab')
    p.add_argument('-trbsz', '--train_batch_size', type=int, default=64, help='Training batch size')
    p.add_argument('-dvbsz', '--dev_batch_size', type=int, default=1, help='Dev/Test batch size')
    p.add_argument('-embsz', '--embedding_size', type=int, help='Embedding dimension', required=True)
    p.add_argument('-hidsz', '--hidden_size', type=int, help='Hidden size', required=True)
    p.add_argument('-nenc', '--n_layers_enc', type=int, default=2, help='Num layers in encoder')
    p.add_argument('-ndec', '--n_layers_dec', type=int, default=2, help='Num layers in decoder')
    p.add_argument('-maxlen', '--max_length', type=int, default=50, help='Max length of sequence')
    p.add_argument('-lr', '--lr', type=float, default=0.0001, help='Learning rate')
    p.add_argument('-gc', '--grad_clip', type=float, default=5.0, help='Gradient clipping threshold')
    p.add_argument('-tf', '--teacher_forcing_ratio', type=float, default=1.0, help='Teacher forcing ratio') 
    p.add_argument('-attention', '--attention', type=str, default=None, help='Attention applied to Seq2Seq')
    p.add_argument('-svoc_size', '--src_vocab_size', type=int, default=30000, help='Source vocab size')
    p.add_argument('-tvoc_size', '--tgt_vocab_size', type=int, default=30000, help='Target vocab size')
    p.add_argument('-nheads', '--num_heads', type=int, default=8, help='Number of transformer attention heads')
    p.add_argument('-inpdrop', '--input_dropout', type=float, default=0.1, help='Input dropout')
    p.add_argument('-laydrop', '--layer_dropout', type=float, default=0.1, help='Layer dropout')
    p.add_argument('-attndrop', '--attention_dropout', type=float, default=0.1, help='Transfomer attention dropout')
    p.add_argument('-reldrop', '--relu_dropout', type=float, default=0.1, help='Transformer relu dropout')
    p.add_argument('-decweightshare', '--tgt_emb_prj_weight_sharing', type=bool, default=False, help='Whether to weight tie decoder embed/proj weights')
    p.add_argument('-encdecweightshare', '--emb_src_tgt_weight_sharing', type=bool, default=False, help='Whether to weight tie encoder and decoder weights')
    p.add_argument('-labsmooth', '--label_smoothing', type=float, default=0.1, help='Label smoothing rate')
    p.add_argument('-dff', '--d_ff', type=int, default=2048, help='Size of intermiediate hidden layer in Positionwise Feed Forward Net')
    p.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000, help='Number of warmup steps for learning rate')
    p.add_argument('-exp_name', '--experiment_name', type=str, help='Experiment name', required=True)
    p.add_argument('-model_type', '--model_type', type=str, help='Type of model', required=True)

    args = p.parse_args()

    dir_name = './experiments/' + args.experiment_name

    common_params = ['epochs', 'min_freq', 'train_batch_size', 'dev_batch_size', 'embedding_size', 'n_layers_enc', 
                    'n_layers_dec', 'max_length', 'lr', 'grad_clip', 'teacher_forcing_ratio', 'input_dropout',
                    'layer_dropout', 'tgt_emb_prj_weight_sharing', 'emb_src_tgt_weight_sharing', 'exp_name', 'model_type']

    model2args = {
        'GRU': common_params + ['hidden_size', 'attention'],
        'Transformer': common_params + ['src_vocab_size', 'tgt_vocab_size', 'hidden_size', 'num_heads',
                                        'attention_dropout', 'relu_dropout', 'label_smoothing',
                                        'n_warmup_steps']
    }
    
    try:
        os.mkdir(dir_name)
        print('Writing configuration parameters to: {}'.format(dir_name))
    except FileExistsError:
        print('Directory', dir_name, 'already exists. Rewriting configuration parameters.')
    finally:
        write_params(args, dir_name, model2args)