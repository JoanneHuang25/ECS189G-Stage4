# code/stage_4_code/Dataset_Loader.py
import os
import re
import pickle
import gzip
from collections import Counter
from typing import List

import torch
from torch.utils.data import Dataset # PyTorch Dataset is used directly
import numpy as np

from code.base_class.dataset import dataset # Assuming this base class exists

# ------------------------------
# Tokenization and cleaning
# ------------------------------
TOKEN_WORD = re.compile(r"[a-z']+")
TOKEN_WORD_PUNC = re.compile(r"[a-z']+|[?.!]")
STOP = set("""a an the of to and in is are was were be been for on with that as by this it at from or but not no""".split())

def clean(text: str, *, remove_stop: bool = True, keep_punc: bool = False):
    """Tokenise, optionally drop stop-words, optionally keep punctuation tokens."""
    regex = TOKEN_WORD_PUNC if keep_punc else TOKEN_WORD
    toks = regex.findall(text.lower())
    return [t for t in toks if (t not in STOP) or not remove_stop]

# ------------------------------
# Vocabulary
# ------------------------------
class Vocab:
    PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
    def __init__(self, corpus: List[List[str]], min_freq=2):
        freq = Counter(t for sent in corpus for t in sent)
        self.itos = [self.PAD, self.UNK, self.BOS, self.EOS] + [t for t,c in freq.items() if c>=min_freq]
        self.stoi = {t:i for i,t in enumerate(self.itos)}

    def encode(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi[self.UNK]) for t in toks]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    @property
    def size(self) -> int:
        return len(self.itos)

    def load_pretrained_vocab(self, itos_list: List[str]):
        """Loads a pre-existing vocabulary."""
        self.itos = itos_list
        self.stoi = {t: i for i, t in enumerate(self.itos)}


class Dataset_Loader_IMDB(dataset, Dataset): # Inherits from PyTorch Dataset too
    def __init__(self, dName=None, dDescription=None, source_file_path=None, split='train', vocab=None, max_len=300):
        dataset.__init__(self, dName, dDescription)
        self.dataset_source_file_name = source_file_path # Full path to the pkl.gz file
        self.split = split
        self.max_len = max_len
        self.external_vocab = vocab # To pass train vocab to test
        self.X_encoded = []
        self.y_encoded = []
        self.vocab = None

    def load(self):
        print(f'Loading IMDb {self.split} data from: {self.dataset_source_file_name}')
        if not os.path.exists(self.dataset_source_file_name):
            raise FileNotFoundError(f"IMDb data file not found: {self.dataset_source_file_name}")

        with gzip.open(self.dataset_source_file_name, "rb") as f:
            packed = pickle.load(f)

        if self.external_vocab:
            self.vocab = self.external_vocab
        else:
            self.vocab = Vocab([], min_freq=0) # Placeholder, actual vocab loaded from pkl
            self.vocab.load_pretrained_vocab(packed["vocab"]["itos"])

        docs, labels_raw = [], []
        for lab_name, lab_val in (("pos", 1), ("neg", 0)):
            if self.split in packed["data"] and lab_name in packed["data"][self.split]:
                for toks in packed["data"][self.split][lab_name]:
                    docs.append(toks)
                    labels_raw.append(lab_val)
            else:
                print(f"Warning: {self.split}/{lab_name} not found in packed data.")


        for toks, lab in zip(docs, labels_raw):
            ids = [self.vocab.stoi[self.vocab.BOS]] + self.vocab.encode(toks)[:self.max_len-2] + [self.vocab.stoi[self.vocab.EOS]]
            ids += [self.vocab.stoi[self.vocab.PAD]] * (self.max_len - len(ids))
            self.X_encoded.append(torch.tensor(ids))
            self.y_encoded.append(lab) # Store as int, will be tensored in __getitem__

        print(f'Loaded {len(self.X_encoded)} instances for IMDb {self.split}. Vocab size: {self.vocab.size}')
        # The format expected by script_rnn.py is a direct use of Dataset with DataLoader
        # So, we don't return a dict here, but the instance itself is the dataset
        return self # Or potentially return self.X_encoded, self.y_encoded, self.vocab

    def __len__(self):
        return len(self.X_encoded)

    def __getitem__(self, i):
        return self.X_encoded[i], torch.tensor(self.y_encoded[i], dtype=torch.long)


class Dataset_Loader_Jokes(dataset, Dataset): # Inherits from PyTorch Dataset too
    def __init__(self, dName=None, dDescription=None, source_file_path=None, vocab=None, max_len=40):
        dataset.__init__(self, dName, dDescription)
        self.dataset_source_file_name = source_file_path # Full path to the joke txt file
        self.max_len = max_len
        self.external_vocab = vocab
        self.X_encoded_pairs = []
        self.vocab = None

    def load(self):
        print(f'Loading Joke data from: {self.dataset_source_file_name}')
        if not os.path.exists(self.dataset_source_file_name):
            raise FileNotFoundError(f"Joke data file not found: {self.dataset_source_file_name}")

        sents_cleaned = []
        with open(self.dataset_source_file_name, encoding='utf8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    sents_cleaned.append(clean(line.strip(), remove_stop=False, keep_punc=True))

        if self.external_vocab:
            self.vocab = self.external_vocab
        else:
            self.vocab = Vocab(sents_cleaned, min_freq=1)

        for toks in sents_cleaned:
            ids = [self.vocab.stoi[self.vocab.BOS]] + self.vocab.encode(toks)[:self.max_len-2] + [self.vocab.stoi[self.vocab.EOS]]
            ids += [self.vocab.stoi[self.vocab.PAD]] * (self.max_len - len(ids))
            ids_tensor = torch.tensor(ids)
            self.X_encoded_pairs.append((ids_tensor[:-1], ids_tensor[1:])) # input, target pairs

        print(f'Loaded {len(self.X_encoded_pairs)} joke instances. Vocab size: {self.vocab.size}')
        return self # Or self.X_encoded_pairs, self.vocab

    def __len__(self):
        return len(self.X_encoded_pairs)

    def __getitem__(self, i):
        return self.X_encoded_pairs[i]