from pathlib import Path

import pickle
from keras.models import load_model
def load_experiment(filename):
    root = Path(filename)
    model = load_model(root / 'model.keras')

    with open(root / 'history.pkl', "rb") as file_pi:
        history = pickle.load(file_pi)
    return model, history