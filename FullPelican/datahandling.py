# see https://github.com/bb511/deepsets_synth/
import os
from pathlib import Path
import wget
import tarfile


def download_data():
    dataloader = DataLoader()

    # Precaution: Don't dowload if already downloaded
    if dataloader.raw_dir_nonempty():
        print("Data already downloaded.. ")
    else:
        print("Downloading data...")
        dataloader.download()

    return dataloader.load_data()

class DataLoader:
    def __init__(self, datasuffix='data') -> None:
        self.datasuffix = datasuffix

        cwd = Path.cwd()
        self.datadir = cwd / self.datasuffix
        if not os.path.isdir(self.datadir):
            os.makedirs(self.datadir)

        self.raw_dir = self.datadir / "raw"
        if not self.raw_dir.is_dir():
            os.makedirs(self.raw_dir)

        self.output_files = {}

    def raw_dir_nonempty(self):
        return len([path for path in self.raw_dir.iterdir()]) > 0

    def download(self):
        train_url = (
            "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz"
        )
        test_url = (
            "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_val.tar.gz"
        )
        paths = [(train_url, 'train'), (test_url, 'test')]

        self.output_files = {}

        for data_file_path, name in paths:
            print(f"Loading: {data_file_path} : {name}")
            data_file_path = wget.download(data_file_path, out=str(self.raw_dir))
            data_tar = tarfile.open(data_file_path, "r:gz")
            data_tar.extractall(str(self.raw_dir))
            data_tar.close()
            os.remove(data_file_path)

            self.output_files[name] = data_file_path


    def load_data(self):
        try:
            traindir = self.output_files['train'].iterdir()
            testdir = self.output_files['test'].iterdir()
            return traindir, testdir
        except KeyError as key:
            paths = [path for path in self.raw_dir.iterdir()]

            if len(paths) != 2:
                print(f'Key {key} not found. Consider downloading.')

            for path in paths:
                if 'train' in str(path):
                    self.output_files['train'] = path
                elif 'val' in str(path):
                    self.output_files['test'] = path
            return self.load_data()


def main():
    data = DataLoader("zenodo_data")
    data.download()

if __name__ == '__main__':
    main()
