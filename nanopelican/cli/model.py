from nanopelican.models import PelicanNano


def make_model_from_args(args):
    model = PelicanNano(
        hidden=args.n_hidden,
        outputs=args.n_outputs,
        activation=args.activation,
        data_format=args.data_format,
        dropout=args.dropout_rate,
        batchnorm = args.use_batchnorm,
        num_average_particles = args.num_particles_avg,
        shape=(args.num_particles, args.num_particles, 1)
    )



    return model


