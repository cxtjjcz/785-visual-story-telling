import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
import torchvision.models as models
from hyperparams import *


class fc7_Extractor(nn.Module):
    def __init__(self, fine_tune=False):
        super(fc7_Extractor, self).__init__()
        self.pretrained = models.vgg16(pretrained=True)
        self.fine_tune(fine_tune)

    def forward(self, x):
        x = self.pretrained.features(x)
        x = self.pretrained.avgpool(x)
        x = torch.flatten(x, 1)
        x = nn.Sequential(*list(self.pretrained.classifier.children())[:-1])(x)
        return x

    def fine_tune(self, fine_tune):
        if not fine_tune:
            for p in self.pretrained.parameters():
                p.requires_grad = False


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc7 = fc7_Extractor()
        self.gru = nn.GRU(FC7_SIZE, HIDDEN_SIZE)

    # batch * 5 * 3 * w * h
    def forward(self, images, hidden):
        batch_size, num_pics, channels, width, height = images.size()
        embedded = torch.zeros((num_pics, batch_size, FC7_SIZE))
        for i in range(num_pics):
            batch_i = images[:, -(i+1), :, :, :]  # ith pics
            features = self.fc7(batch_i)  # out shape:batch * 5 * 4096
            embedded[i, :, :] = features  # shape: num_pic * batch * 4096
        output, hidden = self.gru(embedded, hidden)
        # output: num_pic, batch, 1000
        # hidden: 1, batch, 1000
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.gru = nn.GRU(EMBEDDING_SIZE, HIDDEN_SIZE)

    def forward(self, padded_stories, hidden, lens):
        padded_stories = self.embedding(padded_stories)
        padded_stories = pack_padded_sequence(padded_stories, lens, enforce_sorted=False)
        output, hidden = self.gru(padded_stories, hidden)
        return output, hidden


class BaselineModel(nn.Module):
    def __init__(self, vocab):
        super(BaselineModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size=len(vocab))
        self.vocab = vocab
        self.out_layer = nn.Linear(HIDDEN_SIZE, len(vocab))
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()  # default mean
                                
    def init_hidden(self, batch_size, device):
        return torch.rand(1, batch_size, HIDDEN_SIZE, device=device)

    def get_decoded_output(self, decoder_input, hidden, lens):
        output, hidden = self.decoder(decoder_input, hidden, lens)
        output, _ = pad_packed_sequence(output)
        output = self.out_layer(output)
        # output = output.view(output.size()[0], -1)
        return output, hidden

    def forward(self, images, stories, story_lens, device):
        story_lens = torch.Tensor(story_lens)
        batch_size, _, _, _ ,_ = images.size()
        out, hidden = self.encoder(images, self.init_hidden(batch_size, device))
        out, hidden = self.decoder(stories, hidden, story_lens)
        n_tokens = story_lens.sum() - story_lens.size(0)
        loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        out, out_lens = pad_packed_sequence(out)
        out = self.out_layer(out)
        for i in range(out.size()[0]-1):
            active = i + 1 < story_lens
            loss += criterion(out[i, active,: ], stories[i+1, active])
        loss /= n_tokens
        return -loss
