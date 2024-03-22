from nanopelican.models import PELICANnano
from keras.optimizers import AdamW
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy

def make_model_from_args(args):
    model = PELICANnano(
        hidden=args.n_hidden,
        outputs=args.n_outputs,
        activation=args.activation,
        data_format=args.data_format
    )

    loss = None

    if args.n_outputs > 1:
        loss = CategoricalCrossentropy(from_logits=True)
    else:
        loss = BinaryCrossentropy(from_logits=True)

    model.compile(
        optimizer=AdamW(),
        loss=loss,
        metrics=['acc'],
    )

    return model


