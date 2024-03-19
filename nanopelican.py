from keras.models import Model

class PELICANnano(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)