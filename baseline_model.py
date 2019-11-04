import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
        self.gru = nn.GRU(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True)

    def forward(self, target_sents, hidden):
        batch_size, _ = target_sents.size()
        output = self.embedding(target_sents.type(torch.LongTensor))
        # print(output)
        # print(output.size())
        # output shape : bs * 5 * max_sent_len * embeeding_size
        # output = F.relu(output)
        output, hidden = self.gru(output, hidden)
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

    def get_decoded_output(self, decoder_input, hidden):
        output, hidden = self.decoder(decoder_input, hidden)
        output = output.view(output.size()[0], -1)
        return F.softmax(self.out_layer(output), dim=1), hidden

    def forward(self, images, sents, device):
        # require sents to be of shape batch_size * 5 * MAX_LEN
        batch_size, _, _, _ ,_ = images.size()
        out, hidden = self.encoder(images, self.init_hidden(batch_size, device))
        out, hidden = self.decoder(sents, hidden)
        out_embedding = self.out_layer(out)
        output_loss = self.logSoftmax(out_embedding)  # output for calculating loss
        output_return = F.softmax(out_embedding, dim=1)

        loss = 0.0
        # print("output", out.size())
        # print("sentence", sents.size())
        for i in range(MAX_STORY_LEN):
            output_i = output_loss[:, i]
            sent_i = sents[:, i].type(torch.LongTensor)
            # print(output.size(), sents.size())
            # print(output_i.size(), sent_i.size())
            loss += self.loss(output_i, sent_i)

        # check if this work as intended
        # output = self.out_layer(out).view(-1, len(self.vocab))
        # sents = sents.view(-1).type(torch.LongTensor)
        # score = -self.loss(output, sents)
        # print(output_return)
        # print(output_return.shape)
        return -loss, output_return
