import sys, pdb, os, time
import os.path as osp

import torch
from collections import Counter
from PIL import Image
from PIL import ImageFile
from hyperparams import *

# a vocabulary calss adapted from 11731 assignment 1 starter code
# https://phontron.com/class/mtandseq2seq2019/assignments.html
class Vocabulary():
    def __init__(self, sents, freq_cutoff=1):
        self.w2i = {"<s>": 0, "</s>": 1, "<unk>": 2, "<pad>": 3}
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.unk_id = 2
        self.sents = sents
        self.cutoff = freq_cutoff
        self.build()

    def build(self):
        # Start a counter and only include words that appear frequently.
        # freq_cutoff is to be set to 1, until we have a different tokenization method.
        word_freq = Counter()
        for sent in self.sents:
            word_freq["<s>"] += 1
            for word in sent.split():
                word_freq[word] += 1
            word_freq["</s>"] += 1
        
        # Gather valid words that pass cutoff and add them to the respective dictionaries
        valid_words = [w for w, v in word_freq.items() if v >= self.cutoff]
        valid_words = list(set(valid_words)) # Verifying one of each word
        for word in valid_words:
            if (word not in self.w2i.keys()):
                wid = len(self.w2i)
                self.w2i[word] = wid
                self.i2w[wid] = word

    def __getitem__(self, word):
        # pdb.set_trace()
        return self.w2i.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.w2i

    def __len__(self):
        return len(self.w2i)

    def sent2vec(self, sent, tokenized=False):
        if not tokenized:
            sent = sent.split()
        return torch.tensor([self.w2i.get(w, self.unk_id) for w in sent]).type(torch.LongTensor)

    def vec2sent(self, sent):
        result = [self.i2w[i] for i in sent] # need to add ' ' to dictionary
        return " ".join(result)