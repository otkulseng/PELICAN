from nanopelican.models import PELICANnano
from keras.optimizers import AdamW
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy

def make_model_from_args(args):
    model = PELICANnano(
        hidden=args.n_hidden,
        outputs=args.n_outputs,
        activation=args.activation,
        data_format=args.data_format
    )

    model.compile(
        optimizer=AdamW(),
        loss=BinaryCrossentropy(from_logits=True),
        metrics=[BinaryAccuracy()],
    )

    return model


