from nanopelican import cli
from nanopelican.data import load_dataset
from nanopelican.models import load_experiment

from pathlib import Path
def evaluate_models(args):
    models = []
    exp_dir = Path.cwd() / "experiments"
    for file in exp_dir.iterdir():
        if args.models in file.name:
            print(f'Found matching file {file.name}')
            try:
                model, _ = load_experiment(file)
                models.append((model, file.name))

            except OSError as e:
                print(f"Could not load {file.name}: error {e}")
                continue
    print(f'Total number of models to evaluate: {len(models)}')

    if len(models) > 0:
        data = load_dataset(args.data_dir, args, ['test']).test.batch(128)
    for model, filename in models:
        loss, acc = model.evaluate(data)
        print(f'file: {filename}: loss: {loss} accc: {acc}')


def main():
    print("Evaluating Models...")
    args = cli.test_parser()

    return evaluate_models(args)


if __name__ == '__main__':
    main()
