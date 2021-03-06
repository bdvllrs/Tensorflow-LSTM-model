# LSTM Model implementation

## Usage

Create a `dataset` folder in the working directory. And add the files:
* `sentences.eval`
* `sentences.train`
* `sentences.continuation`
in the `dataset` folder.

Dowload and set a `wordembeddings.word2vec` file at the root folder to use pretrained embbedings.

`main.py` is the training model.
* `main.py` for vanilla LSTM (experiment A)
* `main.py -p` for pretrained embeddings (experiment B)
* `main.py -dp` for pretrained embeddings and down project (experiment C)

`test.py` to restore the model and use it

`generation.py` to generate words from sentences.continuation

## Parameters for the files
* `--workdir` (default, current directory) Specifies the path of the work directory
* `--vocsize` (default, 20000) Size of the vocabulary
* `--num-epochs` (default, 100) Number of epochs
* `--print-every` (default, 10) Value of scalars will be save every print-every loop
* `--lr`, `-l` (default, 0.01) Learning rate
* `--nthreads`, `-t` (default, 2) Number of threads to use
* `--max-to-keep` (default, 1) Number of checkpoint to keep
* `--logfile` (default, `default.log`) Path of the log file
* `--verbose` Set file to verbose
* `--save-every` (default, 100) The value of the network will be saved every
* `--pretrained-embedding`, `-p` Use pretrained embedding
* `--down-project`, `-d` Down project for experiment C

## Dependencies

* numpy
* tensorflow
* pickle
* gensim

