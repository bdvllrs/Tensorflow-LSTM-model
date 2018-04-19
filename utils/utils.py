from gensim import models
import numpy as np
from time import gmtime, strftime
import tensorflow as tf


def log_reset(logfile):
    with open(logfile, 'w') as log:
        log.write('')


def log(*text, logfile=None, is_verbose=False):
    """
    Log text
    :param text: text to log
    :param logfile: file to print the log if given
    :param is_verbose:
    """
    if is_verbose:
        print(*text)
    if logfile is not None:
        with open(logfile, 'a') as log:
            text = map(lambda x: str(x), text)
            log.write(strftime("[%Y-%m-%d %H:%M:%S] ", gmtime()) + ' '.join(text) + '\n')


def word_to_index_transform(word_to_index, data):
    """
    Get a batch of sequences and transform the words into indices
    :param word_to_index: dict of relations between words and indices
    :param data: batch to apply the transformation on.
    """
    def transform_word(word):
        assert word in word_to_index.keys(), "The word {} is not in the vocab".format(word)
        return word_to_index[word]
    return np.array(list(map(lambda sequence: list(map(transform_word, sequence)), data)))


def index_to_word_transform(index_to_word, data):
    """
    Get a batch of sequences and transform the words into indices
    :param index_to_word: dict of relations between words and indices
    :param data: batch to apply the transformation on.
    """
    def transform_word(index):
        assert index in index_to_word.keys(), "The index {} is not in the vocab".format(index)
        return index_to_word[index]
    return np.array(list(map(lambda sequence: list(map(transform_word, sequence)), data)))


def print_batch(index_to_word, batch, ask_for_next=False):
    sentences = index_to_word_transform(index_to_word, batch)
    r = None
    for k, sentence in enumerate(sentences):
        print(" ".join(sentence))
        if ask_for_next and r != 'q':
            r = input("\nEnter for next, q to continue to the next batch\n")


def load_embedding(session, vocab, emb, path, dim_embedding, vocab_size):
    """
      :param session: Tensorflow session object
      :param vocab: A dictionary mapping token strings to vocabulary IDs
      :param emb: Embedding tensor of shape vocabulary_size x dim_embedding
      :param path: Path to embedding file
      :param dim_embedding: Dimensionality of the external embedding.
      :param vocab_size:
    """

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab_size))

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})  # here, embeddings are actually set
