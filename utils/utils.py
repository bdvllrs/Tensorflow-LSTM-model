import numpy as np
from time import gmtime, strftime


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
