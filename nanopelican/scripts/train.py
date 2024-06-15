from nanopelican import data, schedulers
import tensorflow as tf
from keras import callbacks, losses, optimizers, metrics
from tqdm.keras import TqdmCallback
from .util import *
import matplotlib.pyplot as plt

from qkeras.utils import model_save_quantized_weights

from .test import evaluate_model

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


    # best_acc_cb  = callbacks.ModelCheckpoint(
    #     filepath= str(Path(save_dir) / 'best_acc.keras'),
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True,
    #     options=None,
    # )
    # best_loss_cb = callbacks.ModelCheckpoint(
    #     filepath= str(Path(save_dir) / 'best_loss.keras'),
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=True,
    #     options=None,
    # )
    best_acc_cb  = CustomModelCheckpoint(
        filepath= str(Path(save_dir) / 'best_acc.weights.h5'),
        monitor='val_binary_accuracy',
        mode='max',
        weights_only=True
    )
    best_loss_cb = CustomModelCheckpoint(
        filepath= str(Path(save_dir) / 'best_loss.weights.h5'),
        monitor='val_loss',
        mode='min',
        weights_only=True
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        mode='max',
        patience=hps['patience']
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        mode='max',
        patience=hps['patience'] // 2
    )

    model_cbs = [TqdmCallback(verbose=conf['hyperparams']['verbose']),
                 train_log_cb,
                 best_acc_cb, best_loss_cb,
                   early_stopping,
                    #  reduce_lr
                     ]

    # Compilation
    if model.output_shape[-1] > 1:
        loss = losses.CategoricalCrossentropy()
        metric = metrics.CategoricalAccuracy()
    else:
        loss = losses.BinaryCrossentropy(from_logits=True)
        metric = metrics.BinaryAccuracy()

    # learning_rate=hps['lr_init']
    learning_rate = schedulers.LinearWarmupCosineAnnealing(
        epochs=hps['epochs'],
        steps_per_epoch=len(dataset.train),
    )
    optimizer = optimizers.Adam(learning_rate=learning_rate)


    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[metric, get_lr_metric(optimizer)],
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


    model.save_weights(Path(save_dir) / 'model.weights.h5')
    model.save(Path(save_dir) / 'model.keras')

    evaluate_model(save_dir, model)










