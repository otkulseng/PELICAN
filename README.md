# Nano Pelican
Keras implementation of the pytorch code on: https://github.com/abogatskiy/PELICAN-nano.

## Installing
It is set up like a python package, so must be pip installed (run command inside folder containing setup.py)
```
$ C:\users\PELICAN\ pip install .
```
## Running

Example command:
```
$ python runscript.py --n_hidden=1 --n_outputs=1 --print_summary --data_dir=data/sample_data --feature_key=Pmu --label_key=is_signal --data_format=fourvec
```

## Doc
Running help:
```
$ python runscript.py --help
```
will give description of all the different runflags possible.
