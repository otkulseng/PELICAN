import argparse

def new_parser():
    parser = argparse.ArgumentParser("Nano Pelican")

    parser.add_argument('--n_hidden', type=int, required=True, help='Number of channels in the hidden layer')
    parser.add_argument('--n_outputs', type=int, required=True, help='Number of outputs (classification classes)')
    parser.add_argument('--activation', type=str, default='relu', help='Activation after every equivariant block (LinEq2v2 block)')

    parser.add_argument('--data_dir', type=str, default='data/', help='Folder where the dataset is stored')
    parser.add_argument('--data_format', type=str, default='fourvec', help='How the data is formatted')
    parser.add_argument('--feature_key', type=str, help='What key the features of the database are stored under')
    parser.add_argument('--label_key', type=str, help='What key the labels of the database are stored under')

    parser.add_argument('--print_summary', action=argparse.BooleanOptionalAction, help='Prints model summary before execution')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=32, help='Number of epochs')
    parser.add_argument('--validation_size', type=int, default=1000, help='Number of samples in validation to test on on epoch end')

    return parser

def init_args():
    parser = new_parser()
    return parser.parse_args()

