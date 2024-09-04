# Example on how to run the quantized version of the code

In this folder, you should be able to run
```console
python model.py
```

`model.py` contains the python code necessary to specify the model using keras functional API. `config.yml` contains all the necessary hyperparameters.

If `evaluate` in `config.yml` is set to `true`, then the model will be evaluated at the end of training.

All logs are written to a folder specified by `save_dir` in `config.yml`. When training is over, you can run
```console
python extract_weights.py --folder=FOLDER
```
to extract the weights into the c-header file `weights.h`