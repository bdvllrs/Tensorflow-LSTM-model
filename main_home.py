# -*- coding: utf-8 -*-
"""
Project for ETH NLU course.

By `Benjamin DEVILLERS`, `Adrien BENAMIRA` and `Esteban LANTER`
"""

import tensorflow as tf
from utils import DataLoader, log, log_reset, word_to_index_transform
from LSTM import lstm, optimize
# import numpy as np
import time
import os
import argparse
import numpy as np

# import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--workdir", default=os.path.curdir, help="Specifies the path of the work directory")
parser.add_argument("--vocsize", type=int, default=20000, help="Size of the vocabulary")
parser.add_argument("--numepochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--printevery", type=int, default=10, help="Value of scalars will be save every print-every loop")
parser.add_argument("--lr", '-l', type=float, default=0.01, help="Learning rate")
parser.add_argument("--nthreads", '-t', type=int, default=2, help="Number of threads to use")
parser.add_argument("--maxtokeep", type=int, default=1, help="Number of checkpoint to keep")
parser.add_argument("--logfile", default="default.log", help="Path of the log file")
parser.add_argument("--verbose", action="store_true", help="Set file to verbose")
parser.add_argument("--saveevery", type=int, default=100, help="The value of the network will be saved every"
                                                               "save-every loop")
args = parser.parse_args()

max_size = 30  # Max size of the sentences, including  the <bos> and <eos> symbol
vocab_size = 20000  # including symbols
embedding_size = 32  # Size of the embedding
hidden_size = 32
batch_size = 64
# Indixes for the <pad>, <bos>, <eos> and <unk> words
pad_index = 0
bos_index = 1
eos_index = 2
unk_index = 3
num_checkpoints = args.maxtokeep
print_every = args.printevery
save_every = args.saveevery
num_epochs = args.numepochs
workdir = args.workdir
is_verbose = args.verbose

learning_rate = args.lr

logpath = os.path.abspath(os.path.join(workdir, args.logfile))
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

# lstm = LSTM(batch_size, embedding_size, vocab_size, hidden_size, max_size)

x = tf.placeholder(tf.int32, (batch_size, max_size - 1), name="x")
label = tf.placeholder(tf.int32, (batch_size, max_size - 1), name="label")
teacher_forcing = tf.placeholder(tf.bool, (), name="teacher_forcing")

output, softmax_output = lstm(x, label, vocab_size, hidden_size, max_size, batch_size, embedding_size,teacher_forcing)

with tf.Session() as sess:
    onehot = tf.argmax(softmax_output, 2)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer, loss, cross_entropy_out = optimize(output, label, learning_rate)
    perplexity = tf.exp(tf.reduce_sum(cross_entropy_out,1))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('perplexity', perplexity)

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


def printVal(onehot_id, index_to_word):
    for k in range(onehot_id.shape[0]):
        out = [index_to_word.get(i) for i in onehot_id[k, :]]
        out = "".join(out)
        res = ' '.join(out.split())
        print(res)


with tf.Session() as sess:
    print('starting')
    log('starting training', logfile=logpath, is_verbose=is_verbose)
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

    # Get a batch with the dataloader and transfrom it into tokens
    sess.run(tf.global_variables_initializer())
    batches = dataloader_train.get_batches(batch_size, num_epochs=num_epochs)
    for num_batch, batch in enumerate(batches):
        log("starting batch", num_batch, logfile=logpath, is_verbose=is_verbose)
        batch = word_to_index_transform(word_to_index, batch)
        batch_input, batch_target = batch[:, :-1], batch[:, 1:]
        _, loss_out = sess.run([optimizer, loss], {x: batch_input,label: batch_target,teacher_forcing: True})
        print(num_batch)
        print(loss_out)
        if num_batch % print_every == 0:
            batches_eval = dataloader_eval.get_batches(batch_size, num_epochs=1)
            perplexities = []
            for num_batch_eval, batch_eval in enumerate(batches_eval):
                batch_eval = word_to_index_transform(word_to_index, batch_eval)
                batch_eval_input, batch_eval_target = batch_eval[:, :-1], batch_eval[:, 1:]
                loss_out_eval,perplexity_out_eval,onehot_out_eval,cross_entropy_out_eval = sess.run([loss,perplexity,onehot,cross_entropy_out], {x: batch_eval_input,label: batch_eval_target,teacher_forcing: False})
                printVal(onehot_out_eval, index_to_word)
                perplexities=np.concatenate((perplexities,perplexity_out_eval))
            #gen_in = np.stack(batch_size*[np.concatenate(([1],np.repeat(0,max_size-2)))],0)
            gen_in = list(dataloader_eval.get_batches(batch_size, num_epochs=1))[1]
            #print('___________________')
            #print(perplexities)
            #print('___________________')
            for i in range(30):
                onehot_out_gen, softmax_output_gen  = sess.run([onehot,softmax_output], {x: gen_in,label: gen_in,teacher_forcing: False})    
                printVal(onehot_out_gen, index_to_word)
                gen_in = onehot_out_gen
            # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
        if num_batch % save_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=num_batch)
            log("Saved model checkpoint to {}\n".format(path), logfile=logpath, is_verbose=is_verbose)
            
            
            
#            
#for i in range(25):
#    np.stack(batch_size*[np.concatenate(([1],np.repeat(0,max_size-1)))],0)
