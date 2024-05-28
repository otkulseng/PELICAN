from nanopelican import data, schedulers
import tensorflow as tf
from keras import callbacks, losses, optimizers, metrics
from tqdm.keras import TqdmCallback
from .util import *


def train(model, conf):
    # For reproducibility
    tf.keras.utils.set_random_seed(conf['seed'])

    # Logging and save directory
    save_dir = create_directory(conf['save_dir'])
    save_config_file(save_dir / 'config.yml', conf)


    # Model and dataset
    dataset = data.load_dataset(conf['dataset'], keys=['train', 'val'])
    model = model(dataset.train.x_data.shape[1:], conf['model'])
    model.summary(expand_nested=True)

    # Callbacks
    hps = conf['hyperparams']
    train_log_cb = callbacks.CSVLogger(save_dir / 'training.log')
    best_acc_cb  = callbacks.ModelCheckpoint(
        filepath=save_dir / 'best_acc.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    best_loss_cb = callbacks.ModelCheckpoint(
        filepath=save_dir / 'best_loss.keras',
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=hps['patience']
    )

    model_cbs = [TqdmCallback(verbose=conf['hyperparams']['verbose']),
                 train_log_cb, best_acc_cb, best_loss_cb, early_stopping]


    # Compilation
    if model.output_shape[-1] > 1:
        loss = losses.CategoricalCrossentropy()
    else:
        loss = losses.BinaryCrossentropy(from_logits=True)

    optimizer = optimizers.Adam(learning_rate=schedulers.LinearWarmupCosineAnnealing(
        epochs=hps['epochs'],
        steps_per_epoch=len(dataset.train),
    ))

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
    )

    dataset.val.shuffle(keepratio=True)
    try:
        model.fit(
            dataset.train.shuffle(keepratio=True).batch(hps['batch_size']),
            epochs = hps['epochs'],
            validation_data = (dataset.val.x_data[:hps['val_size']], dataset.val.y_data[:hps['val_size']]),
            callbacks=model_cbs,
            verbose=0
        )

    except KeyboardInterrupt:
        print("Keyboard Interrupt! Saving progress")

    model.save(save_dir / 'model.keras')







