import numpy as np
import tensorflow as tf
from os import path, getcwd


class DataLoader:
    """### Data loading and preprocessing

    Let's define the `DataLoader` class. This class will help us to load the dataset and do the corresponding preprocessing.
    Preprocessing appens in the `preprocess_data` method.
    """

    def __init__(self, vocab_size=None, max_size=30, tokenizer=None,
                 transform=None):
        """
        :param filename: path of file. If None, will ask to upload a text file.
        :param tokenizer: function to tokenize the sentences. If None, a default
          tokenizer by the space ` ` caracter will be used.
        :param max_size: maximum size of the sentence. All sentences that are longer
          than max_size will be discarded.
        :param vocab_size: Size of the vocab we will use (if default, the vocab will be
          all the words of the dataset)
        :param transform: some additional transformation(s) to apply to the dataset.
          This can be a callable or a list of collable.
        """
        self.dataset = None
        self.indices = None
        self.max_size = max_size
        self.vocab_size = vocab_size
        self.vocab = None
        self.transform = transform

        if tokenizer is None:
            self.tokenizer = DataLoader.default_tokenizer
        else:
            self.tokenizer = tokenizer

    @staticmethod
    def default_tokenizer(sentence):
        """
        Default tokenizer. Split the sentence by the space.
        :param sentence:
        """
        return sentence.split(' ')

    def pad_sentence(self, words):
        """
        Pad the sentence if the length is lower than self.max_size
        :param words: list of words of the sentence
        """
        nb_of_pad = self.max_size - 2 - len(words)
        words = ['<bos>'] + words + ['<eos>']
        words.extend(['<pad>'] * nb_of_pad)
        return words

    def set_unk_token(self, words):
        """
        Change all unknown words to the <unk> token
        :param words: list of words of the sentence
        """
        vocab = self.get_vocab()
        return list(map(lambda word: word if word in vocab else '<unk>', words))

    def import_file(self, filename):
        """
        Import a file
        """
        dataset_path = path.abspath(path.join(getcwd(), filename))
        # function to remove \n character
        remove_return_character = lambda s: s.replace('\n', '')
        with open(dataset_path, 'r') as f:
            self.dataset = list(map(remove_return_character, f.readlines()))
        return self.preprocess_dataset()

    def preprocess_dataset(self):
        """
        Preprocess the dataset.

        * Tokenize the sentences
        * Remove too long sentences
        * Pad the sentences
        * Set the <unk> token for unknown words
        """
        # Tokenize the dataset
        self.dataset = list(map(self.tokenizer, self.dataset))
        # Remove sentences that are too long.
        # We use self.max_size -2 for <bos> and <eos>
        self.dataset = list(filter(lambda s: len(s) <= (self.max_size - 2),
                                   self.dataset))
        # Set the <unk> words
        self.dataset = list(map(self.set_unk_token, self.dataset))
        # Pad the sentences and add <bos> and <eos>
        self.dataset = list(map(self.pad_sentence, self.dataset))
        # Apply transforms
        if self.transform is not None:
            if type(self.transform) == list:
                for transform in self.transform:
                    self.dataset = transform(self.dataset)
            else:
                self.dataset = self.transform(self.dataset)

        # indices of the dataset
        self.indices = list(range(len(self.dataset)))
        self.dataset = np.array(self.dataset)
        np.random.shuffle(self.indices)

        return self  # For method chaining

    def get_dataset(self, tf_dataset=False):
        """
        Dataset getter
        :param tf_dataset: if True, will give the tf dataset. Otherwise list dataset.
        """
        return tf.data.Dataset.from_tensors(self.dataset) if tf_dataset else self.dataset

    def get_vocab(self):
        """
        Get vocab of the dataset.
        """
        # To be quicker if we have to get the vocab several times
        if self.vocab is not None:
            return self.vocab
        # vocab is a dict where the key is a word and the value the number of ...
        # appearance of the word
        vocab = {}
        for sentence in self.dataset:
            for word in sentence:
                vocab[word] = vocab[word] + 1 if word in vocab.keys() else 1
        # Sort vocab by number of appearance then get the voc
        vocab_ordered = list(zip(*sorted(vocab.items(), key=lambda t: t[1],
                                         reverse=True)))
        vocab_ordered = vocab_ordered[0]  # Keep only the words
        if self.vocab_size is None:
            self.vocab = vocab_ordered
        else:
            self.vocab = vocab_ordered[:self.vocab_size]
        return self.vocab

    def get_word_to_index(self, pad_index=0, bos_index=1, eos_index=2, unk_index=3):
        """
        Build the word to index correspondance
        :param pad_index: index of the padding word
        :param bos_index: index of the bos word
        :param eos_index: index of the eos word
        :param unk_index: index of the unk word
        :rtype: tuple(dict, dict)
        :return: couple of word to index correspondance and index to word correspondance.
        """
        vocab = self.get_vocab()
        word_to_index = {
            "<pad>": pad_index,
            "<bos>": bos_index,
            "<eos>": eos_index,
            "<unk>": unk_index
        }
        index_to_word = {
            pad_index: "<pad>",
            bos_index: "<bos>",
            eos_index: "<eos>",
            unk_index: "<unk>"
        }
        max_already_taken_token = max([pad_index, bos_index, eos_index, unk_index])
        for k, word in enumerate(vocab):
            word_to_index[word] = max_already_taken_token + k + 1
            index_to_word[max_already_taken_token + k + 1] = word
        return word_to_index, index_to_word

    def apply_transformation(self, transform):
        """
        Apply a transformation to the current dataset
        :param transform: function to apply to the dataset.
            The transform function gets a value of the dataset and has to return the new value.
        """
        self.dataset = list(map(transform, self.dataset))
        return self

    def regenerate_indices(self):
        """
        Rebuilds the list of shuffled dataset indices for random batching
        """
        self.indices = list(range(len(self.dataset)))
        np.random.shuffle(self.indices)

    def get_batch(self, batch_size):
        """
        Get a batch of random elements
        """
        # If there is not enough elements, restart from scratch
        if len(self.indices) < batch_size:
            self.regenerate_incices()
        return np.array([self.dataset[self.indices.pop()] for k in range(batch_size)])

