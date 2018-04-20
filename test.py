# -*- coding: utf-8 -*-
"""
Project for ETH NLU course.

Restore the graph
Calculates the perplexity on the training set

By `Benjamin DEVILLERS`, `Adrien BENAMIRA` and `Esteban LANTER`
"""

import tensorflow as tf
from utils import DataLoader, log, log_reset, word_to_index_transform, load_embedding, print_batch
import os
import argparse
import numpy as np

# import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", default=os.path.curdir, help="Specifies the path of the work directory")
parser.add_argument("--vocsize", type=int, default=20000, help="Size of the vocabulary")
parser.add_argument("--num-epochs", "--numepochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--print-every", "--printevery", type=int, default=100,
                    help="Value of scalars will be save every print-every loop")
parser.add_argument("--lr", '-l', type=float, default=0.01, help="Learning rate")
parser.add_argument("--nthreads", '-t', type=int, default=2, help="Number of threads to use")
parser.add_argument("--max-to-keep", "--maxtokeep", type=int, default=1, help="Number of checkpoint to keep")
parser.add_argument("--logfile", default="default.log", help="Path of the log file")
parser.add_argument("--verbose", action="store_true", help="Set file to verbose")
parser.add_argument("--pretrained-embedding", "-p", action="store_true", help="Use pretrained embedding")
parser.add_argument("--save-every", "--saveevery", type=int, default=100,
                    help="The value of the network will be saved every save-every loop")
args = parser.parse_args()

max_size = 30  # Max size of the sentences, including  the <bos> and <eos> symbol
vocab_size = 20000  # including symbols
embedding_size = 100  # Size of the embedding
hidden_size = 100
batch_size = 64
# Indixes for the <pad>, <bos>, <eos> and <unk> words
pad_index = 0
bos_index = 1
eos_index = 2
unk_index = 3
num_checkpoints = args.max_to_keep
print_every = args.print_every
save_every = args.save_every
num_epochs = args.num_epochs
workdir = args.workdir
is_verbose = args.verbose
use_pretrained_model = args.pretrained_embedding

learning_rate = args.lr

logpath = os.path.abspath(os.path.join(workdir, args.logfile))
log_reset(logpath)

"""Loading datasets"""
dataloader_eval = DataLoader('dataset/sentences.eval', vocab_size, max_size, workdir=workdir)

"""Get the vocab and save it to vocab.dat.
If method not called, the model tries to open vocab.dat
Otherwise if no savefile is given, just loads the vocab
directly.
Uncomment to generate the vocab.dat file"""

# dataloader_train.compute_vocab(savefile='vocab.dat')

"""Let's do some test on the dataloader..."""

word_to_index, index_to_word = dataloader_eval.get_word_to_index(pad_index, bos_index,
                                                                 eos_index, unk_index)

nthreads_intra = args.nthreads // 2
nthreads_inter = args.nthreads - args.nthreads // 2

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./run_leonhard/checkpoints/1524156448/model-185200.meta')
    saver.restore(sess, tf.train.latest_checkpoint("./run_leonhard/checkpoints/1524156448/"))

    # sess.run(tf.global_variables_initializer())
    # Output directory for models and summaries

    """Loading pretrained embedding"""
    if use_pretrained_model:
        load_embedding(sess, word_to_index, 'embedding/word_embeddings:0', './wordembeddings.word2vec', embedding_size,
                       vocab_size)

    # Get a batch with the dataloader and transfrom it into tokens
    batches = dataloader_eval.get_batches(batch_size, num_epochs=num_epochs, random=False)
    for num_batch in range(10):
        batch = next(batches)
        # log("starting batch", num_batch, logfile=logpath, is_verbose=is_verbose)
        batch = word_to_index_transform(word_to_index, batch)
        batch_input, batch_target = batch[:, :-1], batch[:, 1:]
        softmax, cross_entropy = sess.run(["softmax_output:0", "optimizer/cross_entropy:0"],
                           {"x:0": batch_input, "label:0": batch_target, "teacher_forcing:0": False})
        onehot = np.argmax(softmax, axis=2)

        print_batch(index_to_word, onehot, ask_for_next=True)
