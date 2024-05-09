#!/usr/bin/env python

import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from nanopelican.data import load_dataset
from nanopelican.models import load_model

import h5py
import pandas as pd

from sklearn.metrics import roc_curve, auc


def load_yaml(filename):
    with open(filename, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    return config



def generate_auc(model, data):
    y_score = model.predict(data.x_data)
    y_true = data.y_data

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    score = auc(fpr, tpr)
    return (fpr, tpr, thresholds, score)

def evaluate_model(model, data):
    loss, acc = model.evaluate(data.batch(1000))
    return loss, acc

def runtime_data(experiment):
    file = pd.read_csv(experiment / 'training.log')
    return file['accuracy'], file['loss'], file['val_accuracy'], file['val_loss']

def run_test(experiment):

    if (experiment / 'evaluation.csv').exists():
        print("evaluation.csv already exist!")
        return

    config = load_yaml(experiment / 'config.yml')
    dataset = load_dataset(config['dataset'], keys=['test']).test


    models_dataframe = []

    with h5py.File(experiment / 'model_metrics.h5', 'w') as save_file:

        acc, loss, val_acc, val_loss = runtime_data(experiment)
        val_group = save_file.create_group('runtime')
        val_group.create_dataset('acc', data=acc)
        val_group.create_dataset('loss', data=loss)
        val_group.create_dataset('val_acc', data=val_acc)
        val_group.create_dataset('val_loss', data=val_loss)

        for file in experiment.iterdir():

            if '.keras' not in file.name[-6:]:
                continue

            try:
                model = load_model(file)
                model_group = save_file.create_group(file.name)



                (loss, acc) = evaluate_model(model, dataset)

                models_dataframe.append(
                    [file.name, acc, auc, loss]
                )

                (fpr, tpr, thresholds, auc) = generate_auc(model, dataset)
                auc_group = model_group.create_group('auc')
                auc_group.create_dataset('fpr', data=fpr)
                auc_group.create_dataset('tpr', data=tpr)
                auc_group.create_dataset('thresholds',  data=thresholds)

                print(f"Done with {file.name}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f'{file.name} failed with error: {e}')
                continue

        df = pd.DataFrame(models_dataframe, columns=['Name', 'acc', 'auc', 'loss'])
        df.to_csv(experiment / 'evaluation.csv', index=False)


def load_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir", required=True,  type=str)

    parser.add_argument("--name",  type=str, default='')

    args = parser.parse_args()
    return args

def get_dictionary_value(dictionary, key):
    for k, v in dictionary.items():
        # print(k)
        if k == key:
            return True, v
        if type(v) == dict:
            found, val = get_dictionary_value(v, key)
            if found:
                return True, val
    return False, None


def generate_dataframe(experiments, col_names):

    frames = []
    for exp in experiments:
        config = load_yaml(exp / 'config.yml')
        test_results = pd.read_csv(exp / 'evaluation.csv')

        for col in col_names:
            found, val = get_dictionary_value(config, col)

            if not found:
                raise AttributeError(f"Could not find attribute {col} in {exp / 'config.yml'}")

            test_results.insert(1, col, [val]*test_results.shape[0])
        frames.append(test_results)
    return pd.concat(frames)



def main():
    # Step 1: Load arguments and find all experiments matching CLI
    args = load_arguments()

    root = Path(args.dir)

    if not root.is_dir():
        raise NotADirectoryError(f"{args.dir} not a directory!")


    experiments = []
    for file in root.iterdir():
        if file.is_dir() and args.name in file.name:
            experiments.append(file)

    print(f'Found {len(experiments)} tests')

    # Step 2: Run all the experiments. Will return if already run
    for exp in tqdm(experiments):
        try:
            run_test(exp)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f'Experiment failed with error {e}')
            continue

    # Step 3: Go through all experiments, create averages etc
    df = generate_dataframe(experiments, ['n_hidden', 'num_particles'])
    filename = Path(args.dir) / f'eval-{args.name}.csv'
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    main()

