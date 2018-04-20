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

"""
Initializing some arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument("--workdir", default=os.path.curdir, help="Specifies the path of the work directory")
parser.add_argument("--vocsize", type=int, default=20000, help="Size of the vocabulary")
parser.add_argument("--logfile", default="default.log", help="Path of the log file")
parser.add_argument("--verbose", action="store_true", help="Set file to verbose")
parser.add_argument("--nthreads", '-t', type=int, default=2, help="Number of threads to use")

args = parser.parse_args()

# Some parameters
max_size = 30  # Max size of the sentences, including  the <bos> and <eos> symbol
vocab_size = args.vocsize  # including symbols
embedding_size = 100  # Size of the embedding
hidden_size = 100
batch_size = 64
# Indixes for the <pad>, <bos>, <eos> and <unk> words
pad_index = 0
bos_index = 1
eos_index = 2
unk_index = 3
workdir = args.workdir
is_verbose = args.verbose

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
    # Restore the model
    saver = tf.train.import_meta_graph('./run_leonhard/checkpoints/1524242137/model-88500.meta')
    saver.restore(sess, tf.train.latest_checkpoint("./run_leonhard/checkpoints/1524242137/"))

    # Output directory for models and summaries

    # Get a batch with the dataloader and transfrom it into tokens
    # Get evaluation sequentialy
    batches = dataloader_eval.get_batches(batch_size, num_epochs=1, random=False, filter_dataset=False)
    last_pos = 0
    print("start writing")
    k = 1
    nlines = 1
    # Reset file
    with open('perplexities.txt', 'w') as file:
        file.write('')

    for batch_item in batches:
        batch = word_to_index_transform(word_to_index, batch_item)
        batch_input, batch_target = batch[:, :-1], batch[:, 1:]
        softmax, cross_entropy = sess.run(["softmax_output:0", "optimizer/cross_entropy:0"],
                                          {"x:0": batch_input, "label:0": batch_target, "teacher_forcing:0": False})
        onehot = np.argmax(softmax, axis=2)
        perplexities = np.exp(cross_entropy, dtype=np.float64)
        nlines += 64
        # print(batch)
        with open('perplexities.txt', 'a') as file:
            file.seek(last_pos)
            for i, perplexity in enumerate(perplexities):
                file.write(str(perplexity) + '\n')
                last_pos = file.tell()
                k += 1
    print(k + len(dataloader_eval.wrong_lines))
