import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from hyperparams import *


class fc7_Extractor(nn.Module):
    def __init__(self):
        super(fc7_Extractor, self).__init__()
        self.pretrained = models.vgg16(pretrained=True)

    def forward(self, x):
        x = self.pretrained.features(x)
        x = self.pretrained.avgpool(x)
        x = torch.flatten(x, 1)
        x = nn.Sequential(*list(self.pretrained.classifier.children())[:-1])(x)
        return x


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
            batch_i = images[:, -i, :, :, :]  # ith pics
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
        self.gru = nn.GRU(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True)

    def forward(self, target_sents, hidden):
        output = self.embedding(target_sents.type(torch.LongTensor))
        # output shape : bs * 5 * max_sent_len * embeeding_size
        # output = F.relu(output)
        output = output.view((BATCH_SIZE, -1, EMBEDDING_SIZE))
        output, hidden = self.gru(output, hidden)
        return output, hidden


class BaselineModel(nn.Module):
    def __init__(self, vocab):
        super(BaselineModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size=len(vocab))
        self.vocab = vocab
        self.out_layer = nn.Linear(HIDDEN_SIZE, len(vocab))
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()  # default mean

    def forward(self, images, sents, hidden):
        # require sents to be of shape batch_size * 5 * MAX_LEN
        out, hidden = self.encoder(images, hidden)
        out, hidden = self.decoder(sents, hidden)
        sents = sents.view(BATCH_SIZE, -1)
        output = self.out_layer(out)
        loss = 0.0
        # print("output", out.size())
        # print("sentence", sents.size())
        for i in range(5 * MAX_SENT_LEN):
            output_i = output[:, i]
            sent_i = sents[:, i].type(torch.LongTensor)
            # print(output.size(), sents.size())
            # print(output_i.size(), sent_i.size())
            loss += self.loss(output_i, sent_i)

        # check if this work as intended
        # output = self.out_layer(out).view(-1, len(self.vocab))
        # sents = sents.view(-1).type(torch.LongTensor)
        # score = -self.loss(output, sents)
        return -loss
