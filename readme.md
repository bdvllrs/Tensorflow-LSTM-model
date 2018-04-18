# LSTM Model implementation

## Usage

Create a `dataset` folder in the working directory. And add the files:
* `sentences.eval`
* `sentences.train`
* `sentences.continuation`
in the `dataset` folder.

## Parameters on the `main.py file`
* `--workdir` (default, current directory) Specifies the path of the work directory
* `--vocsize` (default, 20000) Size of the vocabulary
* `--numepochs` (default, 100) Number of epochs
* `--printevery` (default, 10) Value of scalars will be save every print-every loop
* `--lr` (default, 0.01) Learning rate
* `--nthreads` (default, 2) Number of threads to use
* `--maxtokeep` (default, 1) Number of checkpoint to keep
* `--logfile` (default, `default.log`) Path of the log file
* `--verbose` Set file to verbose
* `--saveevery` (default, 100) The value of the network will be saved every

## Dependencies

* numpy
* tensorflow
* pickle

