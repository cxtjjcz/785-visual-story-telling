# a vocabulary calss adapted from 11731 assignment 1 starter code
# https://phontron.com/class/mtandseq2seq2019/assignments.html
class Vocab():
    def __init__(sents, freq_cutoff=3):
        self.w2i = {"<s>":0, "</s>":1, "<unk>":2}
        self.i2w = {v: k for k, v in self.w2i.items()}

        self.unk_id = 2
        self.sents = sents
        self.cutoff = freq_cutoff
        self.build()

    def build():
        word_freq = Counter(text)
        for sent in self.sents:
            word_freq.add("<s>")
            for word in sent["text"].split():
                word_freq.add(word)
            word_freq.add("</s>")
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
        if word not in self:
            wid = self.w2i[word]
            self.i2w[wid] = word
            return wid
        else:
            return self[word]

    def sent2vec(self, sent, tokenized=False):
        if not tokenized:
            sent = sent.split()
        return np.array([self[w] for w in sent])