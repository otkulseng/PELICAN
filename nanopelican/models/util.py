from pathlib import Path
from nanopelican.schedulers import LinearWarmupCosineAnnealing

import pickle
from keras.models import load_model
from keras import models

def load_model(filename):
    root = Path(filename)
    try:
        model = models.load_model(root, custom_objects={'CosineAnnealingExpDecay': LinearWarmupCosineAnnealing})
    except ValueError:
        model = models.load_model(root)
    return model

def load_history(filename):
    with open(filename, "rb") as file_pi:
        history = pickle.load(file_pi)
    return history