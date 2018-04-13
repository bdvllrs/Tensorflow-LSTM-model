# -*- coding: utf-8 -*-
"""
Project for ETH NLU course.

By `Benjamin DEVILLERS`, `Adrien BENAMIRA` and `Esteban LANTER`
"""

import tensorflow as tf
from utils import DataLoader, word_to_index_transform, index_to_word_transform, lstm, optimize
import numpy as np
import time
import os
import subprocess

# logpath = os.path.abspath(os.path.join(os.path.curdir, "runs"))
# with subprocess.Popen(['tensorboard', '--logdir', logpath]):
#     """Some general settings to edit"""

max_size = 30  # Max size of the sentences, including  the <bos> and <eos> symbol
vocab_size = 100  # including symbols
embedding_size = 100  # Size of the embedding
hidden_size = 512
batch_size = 64
# Indixes for the <pad>, <bos>, <eos> and <unk> words
pad_index = 0
bos_index = 1
eos_index = 2
unk_index = 3
num_checkpoints = 5
print_every = 10
save_every = 100
num_epochs = 100

learning_rate = 0.01

"""Loading datasets"""
dataloader_train = DataLoader('dataset/sentences.train', vocab_size, max_size)
dataloader_eval = DataLoader('dataset/sentences.eval', vocab_size, max_size)

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
print(word_to_index)
print("The index of 'the' is:", word_to_index["the"])
print("The word of index 20 is:", index_to_word[20])

# lstm = LSTM(batch_size, embedding_size, vocab_size, hidden_size, max_size)

x = tf.placeholder(tf.int32, (batch_size, max_size - 1), name="x")
label = tf.placeholder(tf.int32, (batch_size, max_size - 1), name="label")
teacher_forcing = tf.placeholder(tf.bool, (), name="teacher_forcing")

output, softmax_output = lstm(x, label, vocab_size, hidden_size, max_size, batch_size, embedding_size,
                              teacher_forcing)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer, loss = optimize(output, label, learning_rate)
    # perplexity = tf.pow(2, loss)
    tf.summary.scalar('loss', loss)

"""Now let's execute the graph in the session.

We ge a data batch with `dataloader.get_batch(batch_size)`. This fetches a batch of word sequences.

We then need to transform that into a batch of word index. We can achieve this with the helper function
`word_to_index_transform(word_to_index, word_batch)` defined before.

furthermore, we need to seperate the batch into the input batch and the target batch.
We will do that by separating the `max_size - 1` first index of the sequences into the input sequences and
assign the `max_size - 1` last tokens into the target sequences.
"""

with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)) as sess:
    print('starting training')
    sess.run(tf.global_variables_initializer())
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
    # Train Summaries
    train_summary_dir = os.path.join(out_dir, "summaries", timestamp, "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    train_summary_writer.add_graph(sess.graph)
    # Test Summary
    test_summary_dir = os.path.join(out_dir, "summaries", timestamp, "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)
    test_summary_writer.add_graph(sess.graph)

    merged_summary = tf.summary.merge_all()
    print('summary')

    # Get a batch with the dataloader and transfrom it into tokens
    batches = dataloader_train.get_batches(batch_size, num_epochs=num_epochs)
    batches_eval = dataloader_eval.get_batches(batch_size, num_epochs=num_epochs)
    for num_batch, batch in enumerate(batches):
        print("starting batch", num_batch)
        batch = word_to_index_transform(word_to_index, batch)
        # Defining input and target sequences
        batch_input, batch_target = batch[:, :-1], batch[:, 1:]
        # Run the session
        _, logits, out_loss = sess.run([optimizer, softmax_output, loss], {x: batch_input,
                                                                           label: batch_target,
                                                                           teacher_forcing: True})

        if num_batch % print_every == 0:
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

            print("saving scalar")
            test_summary_writer.add_summary(summary_test, num_batch)
            train_summary_writer.add_summary(summary_train, num_batch)

            # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
        if num_batch % save_every == 0:
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints", timestamp))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
            path = saver.save(sess, checkpoint_prefix, global_step=num_batch)
            print("Saved model checkpoint to {}\n".format(path))
