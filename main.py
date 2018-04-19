# -*- coding: utf-8 -*-
"""
Project for ETH NLU course.

By `Benjamin DEVILLERS`, `Adrien BENAMIRA` and `Esteban LANTER`
"""

import tensorflow as tf
from utils import DataLoader, log, log_reset, word_to_index_transform, load_embedding
from LSTM import lstm, optimize
# import numpy as np
import time
import os
import argparse

# import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", default=os.path.curdir, help="Specifies the path of the work directory")
parser.add_argument("--vocsize", type=int, default=20000, help="Size of the vocabulary")
parser.add_argument("--num-epochs", "--numepochs", type=int, default=100000, help="Number of epochs")
parser.add_argument("--print-every", "--printevery", type=int, default=10,
                    help="Value of scalars will be save every print-every loop")
parser.add_argument("--lr", '-l', type=float, default=0.01, help="Learning rate")
parser.add_argument("--nthreads", '-t', type=int, default=2, help="Number of threads to use")
parser.add_argument("--max-to-keep", "--maxtokeep", type=int, default=1, help="Number of checkpoint to keep")
parser.add_argument("--logfile", default="default.log", help="Path of the log file")
parser.add_argument("--verbose", action="store_true", help="Set file to verbose")
parser.add_argument("--pretrained-embedding", "-p", action="store_true", help="Use pretrained embedding")
parser.add_argument("--down-project", "-d", action="store_true", help="Down project")
parser.add_argument("--save-every", "--saveevery", type=int, default=100,
                    help="The value of the network will be saved every save-every loop")
args = parser.parse_args()

max_size = 30  # Max size of the sentences, including  the <bos> and <eos> symbol
vocab_size = args.vocsize  # including symbols
embedding_size = 100  # Size of the embedding
hidden_size = 512
down_project = False

if args.down_project:
    hidden_size = 1024
    down_project = 512

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
embeddingpath = os.path.abspath(os.path.join(workdir, './wordembeddings.word2vec'))
log_reset(logpath)

# logpath = os.path.abspath(os.path.join(workdir, "runs"))
# with subprocess.Popen(['tensorboard', '--logdir', logpath]):

"""Loading datasets"""
dataloader_train = DataLoader('dataset/sentences.train', vocab_size, max_size, workdir=workdir)
dataloader_eval = DataLoader('dataset/sentences.eval', vocab_size, max_size, workdir=workdir)

"""Get the vocab and save it to vocab.dat.
If method not called, the model tries to open vocab.dat
Otherwise if no savefile is given, just loads the vocab
directly.
Uncomment to generate the vocab.dat file"""

# dataloader_train.compute_vocab(savefile='vocab.dat')

"""Let's do some test on the dataloader..."""

# Get the word to index correspondance for the embedding.
word_to_index, index_to_word = dataloader_train.get_word_to_index(pad_index, bos_index,
                                                                  eos_index, unk_index)

log(word_to_index, logfile=logpath, is_verbose=is_verbose)
log("The index of 'the' is:", word_to_index["the"], logfile=logpath, is_verbose=is_verbose)
log("The word of index 20 is:", index_to_word[20], logfile=logpath, is_verbose=is_verbose)

x = tf.placeholder(tf.int32, (batch_size, max_size - 1), name="x")
label = tf.placeholder(tf.int32, (batch_size, max_size - 1), name="label")
teacher_forcing = tf.placeholder(tf.bool, (), name="teacher_forcing")

word_embeddings, output, softmax_output = lstm(x, label, vocab_size, hidden_size, max_size, batch_size, embedding_size,
                                               teacher_forcing, down_project)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer, loss, cross_entropy_out, weights = optimize(output, label, learning_rate)
    perplexity = tf.exp(cross_entropy_out)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('perplexity', tf.reduce_mean(perplexity))

"""Now let's execute the graph in the session.

We ge a data batch with `dataloader.get_batch(batch_size)`. This fetches a batch of word sequences.

We then need to transform that into a batch of word index. We can achieve this with the helper function
`word_to_index_transform(word_to_index, word_batch)` defined before.

furthermore, we need to seperate the batch into the input batch and the target batch.
We will do that by separating the `max_size - 1` first index of the sequences into the input sequences and
assign the `max_size - 1` last tokens into the target sequences.
"""
nthreads_intra = args.nthreads // 2
nthreads_inter = args.nthreads - args.nthreads // 2

with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=nthreads_inter,
                                      intra_op_parallelism_threads=nthreads_intra)) as sess:
    log('starting training', logfile=logpath, is_verbose=is_verbose)
    sess.run(tf.global_variables_initializer())
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(workdir, "runs"))
    # Train Summaries
    train_summary_dir = os.path.join(out_dir, "summaries", timestamp, "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    train_summary_writer.add_graph(sess.graph)
    # Test Summary
    test_summary_dir = os.path.join(out_dir, "summaries", timestamp, "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)
    test_summary_writer.add_graph(sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints", timestamp))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

    merged_summary = tf.summary.merge_all()
    log('summary', logfile=logpath, is_verbose=is_verbose)

    """Loading pretrained embedding"""
    if use_pretrained_model:
        load_embedding(sess, word_to_index, word_embeddings, embeddingpath, embedding_size, vocab_size)

    # Get a batch with the dataloader and transfrom it into tokens
    batches = dataloader_train.get_batches(batch_size, num_epochs=num_epochs)
    batches_eval = dataloader_eval.get_batches(batch_size, num_epochs=num_epochs)
    for num_batch, batch in enumerate(batches):
        log("starting batch", num_batch, logfile=logpath, is_verbose=is_verbose)
        batch = word_to_index_transform(word_to_index, batch)
        # Defining input and target sequences
        batch_input, batch_target = batch[:, :-1], batch[:, 1:]
        # Run the session
        _, logits, out_loss, computed_perplexity = sess.run([optimizer, softmax_output, loss, perplexity],
                                                            {x: batch_input,
                                                             label: batch_target,
                                                             teacher_forcing: True})

        if num_batch % print_every == 0:
            try:
                batch_eval = next(batches_eval)
            except:
                batches_eval = dataloader_eval.get_batches(batch_size, num_epochs=num_epochs)
                batch_eval = next(batches_eval)
            batch_eval = word_to_index_transform(word_to_index, batch_eval)
            # Defining input and target sequences
            batch_eval_input, batch_eval_target = batch_eval[:, :-1], batch_eval[:, 1:]
            summary_test = sess.run(merged_summary, {x: batch_eval_input,
                                                     label: batch_eval_target,
                                                     teacher_forcing: False})
            summary_train = sess.run(merged_summary, {x: batch_input,
                                                      label: batch_target,
                                                      teacher_forcing: True})

            log("saving scalar", logfile=logpath, is_verbose=is_verbose)
            test_summary_writer.add_summary(summary_test, num_batch)
            train_summary_writer.add_summary(summary_train, num_batch)

            # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
        if num_batch % save_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=num_batch)
            log("Saved model checkpoint to {}\n".format(path), logfile=logpath, is_verbose=is_verbose)
