# a vocabulary calss adapted from 11731 assignment 1 starter code
# https://phontron.com/class/mtandseq2seq2019/assignments.html
from collections import Counter
import torch
import pickle

class Vocabulary():
    def __init__(self, sents, freq_cutoff=3):
        self.w2i = {"<s>":0, "</s>":1, "<unk>":2, "<pad>":3}
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.unk_id = 2
        self.sents = sents
        self.cutoff = freq_cutoff
        self.build()

    def build(self):
        word_freq = Counter()
        for sent in self.sents:
            word_freq["<s>"] += 1
            for word in sent.split():
                word_freq[word] += 1
            word_freq["</s>"] += 1
        valid_words = [w for w, v in word_freq.items() if v >= self.cutoff]
        for word in valid_words:
            self.add(word)

    def __getitem__(self, word):
        return self.w2i.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.w2i

    def __len__(self):
        return len(self.w2i)

    def i2w(self, wid):
        return self.i2w[wid]

    def add(self, word):
        if word not in self.w2i:
            wid = self.w2i[word] = len(self.w2i)
            self.i2w[wid] = word
            return wid
        else:
            return self.w2i[word]

    def sent2vec(self, sent, tokenized=False):
        if not tokenized:
            sent = sent.split()
        return torch.tensor([self[w] for w in sent])
