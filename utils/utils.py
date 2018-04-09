import numpy as np


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
