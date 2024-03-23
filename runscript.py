from nanopelican import cli

from tqdm.keras import TqdmCallback

def main():

    args = cli.init_args()

    model = cli.make_model_from_args(args)
    dataset  = cli.make_dataset_from_args(args)

    if args.print_summary:
        for _, data in dataset.items():
            data = data.as_numpy_iterator()
            feature, _ = next(data)
            model.build(feature.shape)
            model.summary()
            break

    model.fit(
        dataset['train'].shuffle(dataset['train'].cardinality()).batch(args.batch_size),
        epochs=args.epochs,
        validation_data=dataset['val'].shuffle(dataset['val'].cardinality()).take(args.validation_size), # Not ideal!!!!!
        callbacks=[TqdmCallback()],
        verbose=0
    )

    model.save_all_to_dir("myones", args)
















if __name__ == '__main__':
    main()
