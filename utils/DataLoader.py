import numpy as np
import tensorflow as tf
import pickle
from os import path, getcwd


class DataLoader:
    """### Data loading and preprocessing

    Let's define the `DataLoader` class. This class will help us to load the dataset and do the corresponding preprocessing.
    Preprocessing appens in the `preprocess_data` method.
    """

    def __init__(self, filename, vocab_size=None, max_size=30, transform=None, workdir=None):
        """
        :param filename: path of file.
        :param vocab_size: Size of the vocab we will use (if default, the vocab will be
          all the words of the dataset)
        :param max_size: maximum size of the sentence. All sentences that are longer
          than max_size will be discarded.
        :param transform: some additional transformation(s) to apply to the dataset.
          This can be a callable or a list of collable.
        """
        if workdir is None:
            workdir = getcwd()
        self.workdir = workdir
        self.filename = path.abspath(path.join(self.workdir, filename))
        self.dataset = None
        self.indices = None
        self.max_size = max_size
        self.vocab_size = vocab_size
        self.vocab = None
        self.transform = transform
        self.tokenizer = lambda x: x.split(' ')
        self.original_lines = None
        self.lines = None
        self.current_epoch = None
        self.wrong_lines = []

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

    def preprocess_dataset(self, batch, filter_dataset=True):
        """
        Preprocess the dataset.

        * Tokenize the sentences
        * Remove too long sentences
        * Pad the sentences
        * Set the <unk> token for unknown words
        """
        # Tokenize the dataset
        dataset = list(map(self.tokenizer, batch))
        # Remove sentences that are too long.
        # We use self.max_size -2 for <bos> and <eos>
        if filter_dataset:
            dataset = list(filter(lambda s: len(s) <= (self.max_size - 2), dataset))
        else:
            dataset = list(map(lambda s: s[:self.max_size - 2], dataset))
        # Set the <unk> words
        dataset = list(map(self.set_unk_token, dataset))
        # Pad the sentences and add <bos> and <eos>
        dataset = list(map(self.pad_sentence, dataset))
        # Apply transforms
        if self.transform is not None:
            if type(self.transform) == list:
                for transform in self.transform:
                    dataset = transform(dataset)
            else:
                dataset = self.transform(dataset)

        # indices of the dataset
        # self.indices = list(range(len(self.dataset)))
        return np.array(dataset)

    def get_vocab(self, file=None):
        """
        Get vocab of the dataset.
        :param file: file of the vocab pickled file
        """
        if self.vocab is not None:
            return self.vocab
        if file is None:
            file = 'vocab.dat'
        file = path.abspath(path.join(self.workdir, file))
        with open(file, 'rb') as file:
            self.vocab = pickle.load(file)[:self.vocab_size - 4]  # Removing the 3 tokens in the size
        return self.vocab

    def compute_vocab(self, vocab_size=20000, savefile=None):
        """
        Compute the vocabulary from the dataset filename
        :param vocab_size: size of the vocab
        :param savefile: to save the file in a file
        :return: vocab
        """
        filepath = path.abspath(path.join(self.workdir, self.filename))
        vocab = {}
        with open(filepath, 'r') as file:
            line = file.readline()
            while line:
                words = line.split(' ')
                for word in words:
                    # Gets rid of the \n symbol
                    if word[-1:] == '\n':
                        word = word[:-1]
                    if word not in vocab.keys():
                        vocab[word] = 1
                    else:
                        vocab[word] += 1
                line = file.readline()
        vocab_ordered = list(zip(*sorted(vocab.items(), key=lambda t: t[1],
                                         reverse=True)))
        self.vocab = vocab_ordered[0]  # Keep only the words
        self.vocab = self.vocab[:vocab_size - 4]  # Remove the <bos>, <unk> and <eos> tokens
        if savefile is not None:
            with open('vocab.dat', 'wb') as file:
                pickle.dump(self.vocab, file)
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
        print(pad_index)
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

    def get_lines(self, random=True, filter_dataset=True):
        """
        Store all the lines
        :return:
        """
        if self.original_lines is None:
            self.original_lines = []
        with open(self.filename, 'r') as dataset:
            self.original_lines.append(dataset.tell())
            line = dataset.readline()
            while line:
                if not filter_dataset or len(line.split(' ')) <= self.max_size - 2:
                    self.original_lines.append(dataset.tell())
                else:
                    wrong_line = dataset.tell()
                    if wrong_line not in self.wrong_lines:
                        self.wrong_lines.append(wrong_line)
                line = dataset.readline()

        print("Importing", len(self.original_lines), "sentences")
        print(len(self.wrong_lines), "lines have not been included because of their size")
        self.reinit_lines(random)

    def reinit_lines(self, random=True):
        self.lines = self.original_lines[:]
        if random:
            np.random.shuffle(self.lines)

    def get_batch(self, batch_size, random=True, filter_dataset=True):
        if self.original_lines is None:
            self.get_lines(random)
        with open(self.filename, 'r') as dataset:
            batch = []
            epoch_changed = False
            while len(batch) < batch_size:
                if not len(self.lines):
                    self.reinit_lines(random)
                    epoch_changed = True
                pos = self.lines.pop(0)
                dataset.seek(pos)
                line = dataset.readline()
                if not filter_dataset or len(line.split(' ')) <= self.max_size - 2:
                    line = line.replace('\n', '')
                    line = line.replace('\t', '')
                    batch.append(line)
                # else:
                #     wrong_line = self.original_lines.index(pos)+1
                #     if wrong_line not in self.wrong_lines:
                #         self.wrong_lines.append(wrong_line)
        return batch, epoch_changed

    def get_batches(self, batch_size, num_epochs, random=True, filter_dataset=True):
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_changed = False
            while not epoch_changed:
                batch, epoch_changed = self.get_batch(batch_size, random, filter_dataset)
                yield self.preprocess_dataset(batch, filter_dataset)

    def get_batches_old(self, batch_size, num_epochs):
        """
        Get a batch of random elements
        :param batch_size:
        :param num_epochs:
        """
        with open(self.filename, 'r') as dataset:
            epoch = 0
            while epoch < num_epochs:
                batch = []
                while len(batch) < batch_size:
                    line = dataset.readline()
                    if not line:  # If no more line, we go back to the beginning
                        dataset.seek(0)
                        epoch += 1
                        line = dataset.readline()
                    # Remove \n symbol
                    line = line.replace('\n', '')
                    if len(line) <= self.max_size:
                        batch.append(line)
                yield self.preprocess_dataset(batch)
