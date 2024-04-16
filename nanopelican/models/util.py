from pathlib import Path
from nanopelican.schedulers import LinearWarmupCosineAnnealing

import pickle
from keras.models import load_model
def load_experiment(filename):
    root = Path(filename)
    try:
        model = load_model(root / 'model.keras', custom_objects={'CosineAnnealingExpDecay': LinearWarmupCosineAnnealing})
    except ValueError:
        model = load_model(root /'model.keras')

    with open(root / 'history.pkl', "rb") as file_pi:
        history = pickle.load(file_pi)
    return model, history