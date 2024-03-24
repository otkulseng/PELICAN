from nanopelican.models import PELICANnano


def make_model_from_args(args):
    model = PELICANnano(
        hidden=args.n_hidden,
        outputs=args.n_outputs,
        activation=args.activation,
        data_format=args.data_format,
        dropout=args.dropout_rate,
        batchnorm = args.use_batchnorm,
        num_average_particles = args.num_particles_avg
    )



    return model


