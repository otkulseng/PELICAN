import argparse

def new_parser():
    parser = argparse.ArgumentParser("Nano Pelican")


    parser.add_argument('--evaluate_models', type=str, help='Evaluates models instead of training')

    parser.add_argument('--n_hidden', type=int, help='Number of channels in the hidden layer')
    parser.add_argument('--n_outputs', type=int, help='Number of outputs (classification classes)')
    parser.add_argument('--activation', type=str, default='relu', help='Activation after every equivariant block (LinEq2v2 block)')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate (default 0)')
    parser.add_argument('--use_batchnorm', action=argparse.BooleanOptionalAction, help='Equips layers with batch normalization')

    parser.add_argument('--data_dir', type=str, default='data/', help='Folder where the dataset is stored')
    parser.add_argument('--data_format', type=str, default='fourvec', help='How the data is formatted')
    parser.add_argument('--feature_key', type=str, help='What key the features of the database are stored under')
    parser.add_argument('--label_key', type=str, help='What key the labels of the database are stored under')
    parser.add_argument('--num_particles', type=int, default=32, help='Number of particles to consider (and make the inner products of)')
    parser.add_argument('--num_particles_avg', type=int, default=1, help='Hyperparameter, should be approx equal to average number of particles')

    parser.add_argument('--print_summary', action=argparse.BooleanOptionalAction, help='Prints model summary before execution')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=32, help='Number of epochs')
    parser.add_argument('--validation_size', type=int, default=1000, help='Number of samples in validation to test on on epoch end')


    parser.add_argument('--experiment_root', type=str, default='experiments', help='Root folder where models and histories are saved')
    parser.add_argument('--experiment_name', type=str, default='experiment', help='Names of the experiments')



    return parser

def init_args():
    parser = new_parser()
    return parser.parse_args()

