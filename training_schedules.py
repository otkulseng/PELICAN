from keras import backend as K
from keras.callbacks import Callback

class TopTagging(Callback):
    def __init__(self):
        super().__init__()