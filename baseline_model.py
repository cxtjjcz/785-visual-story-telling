
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from hyperparams import  *


class fc7_Extractor(nn.Module):
    def __init__(self):
        super(fc7_Extractor, self).__init__()
        self.model = models.vgg16(pretrained=True)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = nn.Sequential(*list(self.model.classifier.children())[:-1])(x)
        return x


# fc7Model(torch.zeros(64,3,32,32)).shape

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder).__init__()
        self.fc7 = fc7_Extractor()
        self.gru = nn.GRU(FC7_SIZE, HIDDEN_SIZE)

    # batch * 5 * 3 * w * h
    def forward(self, images, hidden):
        batch_size, num_pics, channels, width, height = images.size()
        embedded = torch.zeros((num_pics, batch_size,  FC7_SIZE))
        for i in range(num_pics):
            batch_i = images[:, -i, :, :, :] # ith pics
            features = self.fc7(batch_i) # out shape:batch * 5 * 4096
            embedded[i,:,:] = features # shape: num_pic * batch * 4096
        output, hidden = self.gru(embedded, hidden)
        # output: num_pic, batch, 1000
        # hidden: 1, batch, 1000
        return output, hidden

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.gru = nn.GRU(EMBEDDING_SIZE, HIDDEN_SIZE)

    def forward(self, target_sents, hidden):
        output = self.embedding(target_sents)
        output = output.view(1, BATCH_SIZE, -1)
        # output = F.relu(output)
        output, hidden= self.gru(output, hidden)
        return output, hidden

class BaselineModel(nn.Module):
    def __init__(self, vocab):
        super(BaselineModel).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.vocab = vocab
        self.out_layer = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss() # default mean

    def forward(self, images, sents, hidden):
        sents_feats = [self.vocab.sent2vec(s) for s in sents]
        sents_feats = torch.stack(sents_feats)
        out, hidden = self.encoder(images, hidden)
        out, hidden = self.decoder(sents_feats, hidden)
        for i, sent_out in enumerate(out):
            output = self.out_layer(sent_out)
            sm_out = self.softmax(self.out(out[0]))
            loss = self.loss(sm_out, sents_feats[i])
        return loss
