from nanopelican import cli
from keras.models import Sequential
from keras.layers import Flatten, Dense

from tqdm.keras import TqdmCallback
from nanopelican.schedulers import LinearWarmupCosineAnnealing
from keras.optimizers import AdamW
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from nanopelican.models import load_experiment
from pathlib import Path

import tensorflow as tf

from nanopelican.models import PelicanNano

def run_training(args):
    dataset  = cli.make_dataset_from_args(args, filehandlers=['train', 'val'])

    dataset['train'].shuffle().batch(args.batch_size)
    dataset['val'].shuffle().batch(args.batch_size)

    model = PelicanNano(cli_args=vars(args))

    if args.print_summary:
        data = list(dataset.values())[0]
        features, _ = data[0]
        model.build(features.shape)
        model.summary()

    loss = None
    if args.n_outputs > 1:
        loss = CategoricalCrossentropy(from_logits=True)
    else:
        loss = BinaryCrossentropy(from_logits=True)

    model.compile(
        optimizer=AdamW(learning_rate=LinearWarmupCosineAnnealing(
            epochs=args.epochs,
            steps_per_epoch=len(dataset['train']),

        ), weight_decay=0.005),
        # optimizer=AdamW( weight_decay=0.005),
        loss=loss,
        metrics=['acc'],
    )

    model.fit(
        dataset['train'],
        epochs=args.epochs,
        validation_data=dataset['val'],
        callbacks=[TqdmCallback()],
        verbose=0,
    )

    model.save_all_to_dir(args)


def evaluate_models(args):
    models = []
    exp_dir = Path.cwd() / "experiments"
    for file in exp_dir.iterdir():
        if args.evaluate_models in file.name:
            print(f'Found matching file {file.name}')
            try:
                model, _ = load_experiment(file)
                models.append((model, file.name))

            except OSError as e:
                print(f"Could not load {file.name}: error {e}")
                continue
    print(f'Total number of models to evaluate: {len(models)}')

    if len(models) > 0:
        data = cli.make_dataset_from_args(args, filehandlers=['test'])['test'].shuffle().batch(args.batch_size)
    for model, filename in models:
        loss, acc = model.evaluate(data)
        print(f'file: {filename}: loss: {loss} accc: {acc}')


def main():
    tf.get_logger().setLevel('INFO')
    args = cli.init_args()

    if args.evaluate_models is None:
        print("Running Models...")
        return run_training(args)
    print("Evaluating Models...")

    return evaluate_models(args)


if __name__ == '__main__':
    main()
