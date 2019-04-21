import argparse
import os

def write_params(args, dir_name):
    f = open(dir_name + '/params.json', 'w+',)
    f.write('{\n')
    for idx, arg in enumerate(vars(args)):
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
    p.add_argument('-trbsz', '--train_batch_size', type=int, default=32, help='Training batch size')
    p.add_argument('-dvbsz', '--dev_batch_size', type=int, default=1, help='Dev/Test batch size')
    p.add_argument('-embsz', '--embed_size', type=int, help='Embedding dimension', required=True)
    p.add_argument('-hidsz', '--hidden_size', type=int, help='Hidden size', required=True)
    p.add_argument('-nenc', '--n_layers_enc', type=int, default=1, help='Num layers in encoder')
    p.add_argument('-ndec', '--n_layers_dec', type=int, default=1, help='Num layers in decoder')
    p.add_argument('-lr', '--lr', type=float, default=.001, help='Learning rate')
    p.add_argument('-gc', '--grad_clip', type=float, default=5.0, help='Gradient clipping threshold')
    p.add_argument('-tf', '--teacher_forcing_ratio', type=float, default=0.5, help='Teacher forcing ratio')
    p.add_argument('-encdrop_inp', '--input_dropout_p_enc', type=float, default=0.2, help='Dropout for encoder input')
    p.add_argument('-decdrop_inp', '--input_dropout_p_dec', type=float, default=0.2, help='Dropout for decoder input')
    p.add_argument('-drop', '--dropout_p', type=float, default=0.5, help='Dropout probability')
    p.add_argument('-attention', '--attention', type=str, default="None", help="Attention applied to Seq2Seq")
    p.add_argument('-exp_name', '--experiment_name', type=str, help='Experiment name', required=True)
    p.add_argument('-model_type', '--model_type', type=str, help='Type of model', required=True)

    args = p.parse_args()

    dir_name = './experiments/' + args.experiment_name
    
    try:
        os.mkdir(dir_name)
        print('Writing configuration parameters to: {}'.format(dir_name))
    except FileExistsError:
        print('Directory', dir_name, 'already exists. Rewriting configuration parameters.')
    finally:
        write_params(args, dir_name)



