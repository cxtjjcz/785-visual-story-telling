FC7_SIZE = 4096
ENCODER_HIDDEN = 1000
DECODER_HIDDEN = 1000
EMBEDDING_SIZE = 250
VOCAB_SIZE = 1000
NUM_PICS = 5 # www 

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
        self.gru = nn.GRU(FC7_SIZE, ENCODER_HIDDEN)

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
        self.hidden_size = DECODER_HIDDEN
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.gru = nn.GRU(EMBEDDING_SIZE, DECODER_HIDDEN)
        self.out = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, target_sents, hidden):
        # x: batch x 
        # 
        output = self.embedding(target_sents)
        output = output.view(1, BATCH_SIZE, -1)
        # output = F.relu(output)
        output, hidden= self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()        
        self.loss = nn.NLLLoss() # default mean

    # TODO
    def forward(self, x, hidden):
        fc7_feats = self.fc7(x) # batch x 4096
        # out, hidden = self.encoder(fc7_feats, hidden)
        
        return 0

# TODO 