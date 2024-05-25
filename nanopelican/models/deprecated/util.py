from pathlib import Path
from nanopelican.schedulers import LinearWarmupCosineAnnealing

import pickle
import tensorflow as tf

def load_model(filename):
    root = Path(filename)
    try:
        model = tf.keras.models.load_model(root, custom_objects={'CosineAnnealingExpDecay': LinearWarmupCosineAnnealing})
    except ValueError:
        model = tf.keras.models.load_model(root)
    return model

def load_history(filename):
    with open(filename, "rb") as file_pi:
        history = pickle.load(file_pi)
    return history