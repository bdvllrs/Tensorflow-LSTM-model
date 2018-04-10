# -*- coding: utf-8 -*-
"""
Project for ETH NLU course.

By `Benjamin DEVILLERS`, `Adrien BENAMIRA` and `Esteban LANTER`
"""

import tensorflow as tf
from utils import DataLoader, word_to_index_transform
from LSTM import LSTM

"""### Some general settings"""

max_size = 30  # Max size of the sentences, including  the <bos> and <eos> symbol
vocab_size = 20000  # including symbols
embedding_size = 100  # Size of the embeddig
hidden_size = 512
batch_size = 64
# Indixes for the <pad>, <bos>, <eos> and <unk> words
pad_index = 0
bos_index = 1
eos_index = 2
unk_index = 3

learning_rate = 0.01

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
print("The word of index 100 is:", index_to_word[100])

lstm = LSTM(batch_size, embedding_size, vocab_size, hidden_size, max_size)

x = tf.placeholder(tf.int32, (batch_size, max_size - 1), name="x")
label = tf.placeholder(tf.int32, (batch_size, max_size - 1), name="label")

output, softmax_output = lstm(x, label)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = LSTM.optimize(output, label, learning_rate)

batches = dataloader_eval.get_batches(batch_size, 1)
for batch in batches:
    b = word_to_index_transform(word_to_index, batch)
    print(b)
    break

"""Now let's execute the graph in the session.

We ge a data batch with `dataloader.get_batch(batch_size)`. This fetches a batch of word sequences.

We then need to transform that into a batch of word index. We can achieve this with the helper function
`word_to_index_transform(word_to_index, word_batch)` defined before.

furthermore, we need to seperate the batch into the input batch and the target batch.
We will do that by separating the `max_size - 1` first index of the sequences into the input sequences and
assign the `max_size - 1` last tokens into the target sequences.
"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Get a batch with the dataloader and transfrom it into tokens
    batches = dataloader_eval.get_batches(batch_size, num_epochs=1)
    for batch in batches:
        batch = word_to_index_transform(word_to_index, batch)
        # Defining input and target sequences
        batch_input, batch_target = batch[:, :-1], batch[:, 1:]
        # Run the session
        result = sess.run(softmax_output, {x: batch_input, label: batch_target})
        break

"""Train"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batches = dataloader_train.get_batches(batch_size, 100000)
    for batch in batches:
        # Get a batch with the dataloader and transfrom it into tokens
        batch = word_to_index_transform(word_to_index, batch)
        # Defining input and target sequences
        batch_input, batch_target = batch[:, :-1], batch[:, 1:]
        # Run the session
        sess.run(optimizer, {x: batch_input, label: batch_target})
