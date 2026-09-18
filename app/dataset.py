"""
Creates a character-level language-modelling dataset from a stream of text.
You shouldn't need to make any changes to this file.
"""

import torch
from torch.utils.data import Dataset
import re


class CharDataset(Dataset):
    """
    Emits batches of characters

    Each item is a pair ``(x, y)`` of LongTensors of length ``block_size``,
    where ``y`` is ``x`` shifted one character to the right. The vocabulary is
    the sorted set of unique characters in ``data``.

    Attributes:
        stoi: Mapping from character to integer index.
        itos: Mapping from integer index back to character.
        vocab_size: Number of unique characters.
    """

    def __init__(self, config, data):
        """
        Args:
            config: Model CfgNode; only ``block_size`` is read.
            data: The raw training text as a single string.
        """
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        """Return the number of unique characters in the vocabulary."""
        return self.vocab_size

    def get_block_size(self):
        """Return the context length (number of input characters per example)."""
        return self.config.block_size

    def __len__(self):
        """Return the number of ``block_size + 1`` windows in the text."""
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        """
        Return the ``(input, target)`` pair for the window starting at ``idx``.

        Returns:
            Tuple ``(x, y)`` of LongTensors with shape ``(block_size,)``, where
            ``y[i]`` is the character that follows ``x[i]``.
        """
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class WordDataset(Dataset):
    """
    Word-level counterpart to ``CharDataset``.

    The text is lower-cased and split into word tokens (``\\b\\w+\\b``, so
    punctuation is dropped). A ``<UNK>`` token is added to the vocabulary.
    Items have the same ``(x, y)`` shifted-by-one format as ``CharDataset``.
    Not currently used by ``app.main``.
    """

    def __init__(self, config, data):
        """
        Args:
            config: Model CfgNode; only ``block_size`` is read.
            data: The raw training text as a single string.
        """
        # tokenize the text data into words
        self.tokens = re.findall(r"\b\w+\b", data.lower())

        # add '<UNK>' token to the set of words
        words = sorted(set(self.tokens + ["<UNK>"]))

        data_size, vocab_size = len(self.tokens), len(words)
        print("data has %d words, %d unique." % (data_size, vocab_size))

        self.stoi = {w: i for i, w in enumerate(words)}
        self.itos = {i: w for i, w in enumerate(words)}
        self.vocab_size = vocab_size
        self.config = config

    def get_vocab_size(self):
        """Return the number of unique words, including ``<UNK>``."""
        return self.vocab_size

    def get_block_size(self):
        """Return the context length (number of input words per example)."""
        return self.config.block_size

    def __len__(self):
        """Return the number of ``block_size + 1`` windows in the token list."""
        return len(self.tokens) - self.config.block_size

    def __getitem__(self, idx):
        """
        Return the ``(input, target)`` pair for the window starting at ``idx``.

        Returns:
            Tuple ``(x, y)`` of LongTensors with shape ``(block_size,)``.
        """
        # grab a chunk of (block_size + 1) words from the tokens
        chunk = self.tokens[idx : idx + self.config.block_size + 1]
        # encode every word to an integer
        dix = [self.stoi[w] for w in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
