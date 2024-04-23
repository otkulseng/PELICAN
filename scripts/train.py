# Inspired by

import keras
from nanopelican.data import load_dataset
from nanopelican import cli

from tqdm.keras import TqdmCallback
from nanopelican.schedulers import LinearWarmupCosineAnnealing
from keras.optimizers import AdamW, Adam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy

from nanopelican.models import PelicanNano



def run_training(args):
    model = PelicanNano(cli_args=vars(args))

    dataset  = load_dataset(args.data_dir, args, ['train', 'val'])
    dataset.train.shuffle().batch(args.batch_size)
    dataset.val.shuffle()


    if args.print_summary:
        data = dataset.train
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
            steps_per_epoch=len(dataset.train),
        ), weight_decay=0.005),
        loss=loss,
        metrics=['acc'],
    )

    try:
        model.fit(
            dataset.train,
            epochs=args.epochs,
            validation_data=dataset.val,
            callbacks=[TqdmCallback(),
                       keras.callbacks.EarlyStopping(
                    monitor="val_acc",
                    patience=10,
                    verbose=0)
            ],
            verbose=0,
        )
        model.save_all_to_dir(args)


    except KeyboardInterrupt:
        print("Keyboard Interrupt! Saving progress")
        model.save_all_to_dir(args)
        return

def main():
    print("Running Models...")
    args = cli.train_parser()
    return run_training(args)


if __name__ == '__main__':
    main()
