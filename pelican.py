import tensorflow as tf
from layers import Msg, LinEq2v2, LinEq2v0
from keras.layers import Dense, Flatten, Dropout, Softmax
from keras.models import Model

class PELICAN(Model):
    def __init__(self, depth=1, dropout=0.0, activation=None, msg_outputs=10, agg_outputs=10, dense_output=10, scal_outputs=2):
        super().__init__()


        self.msg_layers = [Msg(outputs=msg_outputs, activation=activation) for _ in range(depth)]
        self.dropout    = Dropout(rate=dropout)
        self.agg_layers = [LinEq2v2(outputs=agg_outputs, activation=activation) for _ in range(depth)]
        self.scal_layer = LinEq2v0(outputs=scal_outputs, activation=activation)

        self.dense = Dense(units=dense_output, activation=activation)

    def build(self, input_shape):
        self.l = input_shape[-1]
        self.n = input_shape[-2]


    def call(self, inputs):
        # Note: Assumes input shape to be
        # (Batch) x N x N x L
        # Where N is dimension of 2d input tensors. Every L signals
        # different 2d tensor

        x = inputs
        for msg, agg in zip(self.msg_layers, self.agg_layers):
            x = msg(x)
            x = self.dropout(x)
            x = agg(x)
        x = self.scal_layer(x)
        x = Flatten()(x)
        x = self.dense(x)


        return Softmax()(x)