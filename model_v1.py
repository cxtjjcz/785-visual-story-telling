import sys, pdb, os, time
import os.path as osp

import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from modules import LSTM
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence, pack_padded_sequence
from hyperparams import *

"""
Changes:
    (1) new fc7_extractor
        - uses resnet152 by default
        - added a new final linear layer at the end to project the image feature to vocab embedding dimension
    (2) new encoder and decoder architecture:
        - 5 separate decoders that takes the corresponding image feature (dimension = embedding dimension) 
          and insert it to the front of the sentence embedding vector (as if it's the first word);
          they still use the final hidden state as initial hidden state
          
TODOs:
    - probably a lot of shape errors...
    - unfinished ModelV1 forward function!
    - see TODO comments for more details

"""


class fc7_Extractor(nn.Module):
    def __init__(self, cnn_type, fine_tune=False):
        super(fc7_Extractor, self).__init__()
        self.cnn_type = cnn_type
        if self.cnn_type == "vgg16":
            self.pretrained = models.vgg16(pretrained=True)
            self.fine_tune(fine_tune)
        elif self.cnn_type == "resnet152":
            self.pretrained = models.resnet152(pretrained=True)
            self.fine_tune(fine_tune)

            # overwrite final fc layer to project features to vocab embedding size
            self.pretrained.fc = nn.Linear(self.pretrained.fc.in_features, EMBEDDING_SIZE)
            self.bn = nn.BatchNorm1d(EMBEDDING_SIZE, momentum=0.01)

    def forward(self, x):
        if self.cnn_type == "vgg16":
            x = self.pretrained.features(x)
            x = self.pretrained.avgpool(x)
            x = torch.flatten(x, 1)
            x = nn.Sequential(*list(self.pretrained.classifier.children())[:-1])(x)
        elif "resnet" in self.cnn_type:
            x = self.pretrained(x)  # (batch_size, embed_size)
            x = self.bn(x)
        return x

    def fine_tune(self, fine_tune):
        if not fine_tune:
            for p in self.pretrained.parameters():
                p.requires_grad = False


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # now, fc7_Extractor returns image with dim. of EMBEDDING_SIZE
        self.fc7 = fc7_Extractor(cnn_type=FEATURE_EXTRACTOR)
        # divide hidden size by two when using bidirectional rnn
        if BIDIRECTIONAL_ENCODER:
            hidden_size = HIDDEN_SIZE // 2
        else:
            hidden_size = HIDDEN_SIZE

        self.rnn = LSTM(input_size=EMBEDDING_SIZE, hidden_size=hidden_size,
                        bidirectional=BIDIRECTIONAL_ENCODER, input_drop=INPUT_DROPOUT,
                        output_drop=OUTPUT_DROPOUT, weight_drop=WEIGHT_DROP,
                        num_layers=NUM_LAYERS_ENCODER)

    def forward(self, images, hidden=None):
        """
        :param images: (batch * num_pic * 3 * width * height)
        :param hidden: initial hidden state (default to None)
        :return: image features, encoder outputs, encoder final hidden state
        """
        batch_size, num_pics, channels, width, height = images.size()
        embedded = torch.zeros((num_pics, batch_size, EMBEDDING_SIZE)).to(DEVICE)
        for i in range(num_pics):
            batch_i = images[:, -(i + 1), :, :, :]  # ith pics
            features = self.fc7(batch_i)  # features: batch * embedding_size
            embedded[i, :, :] = features
        output, hidden = self.rnn(embedded, hidden)
        # embedded: num_pic * batch * embedding_size
        # output: num_pic, batch, hidden_size
        # hidden: 1, batch, hidden_size
        return embedded, output, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE, padding_idx=3).to(DEVICE)
        self.rnn = LSTM(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE,
                        bidirectional=BIDIRECTIONAL_DECODER, input_drop=INPUT_DROPOUT,
                        output_drop=OUTPUT_DROPOUT, weight_drop=WEIGHT_DROP,
                        num_layers=NUM_LAYERS_DECODER,
                        batch_first=True)

    def forward(self, image_embedding, padded_sentence, hidden, lens):
        """
        :param image_embedding: image embedding for the corresponding sentence: (batch_size * embedding_size)
        :param padded_sentence: (batch_size * max_seq_len)
        :param hidden: encoder's final hidden state
        :param lens: (batch_size * 1)
        :return:
        """
        padded_sentence = self.embedding(padded_sentence)  # (batch_size * max_seq_len * embedding_size)
        # add image embedding to front (as if it's the first word)
        img_padded_sentence = torch.cat((image_embedding.unsqueeze(1), padded_sentence), dim=1)
        # img_padded_sentence : (batch_size * (max_seq_len+1) * embedding_size)
        img_padded_sentence = img_padded_sentence.permute(1, 0, 2)
        # img_padded_sentence : ((max_seq_len+1) * batch_size * embedding_size)
        lens += 1  # add one to max_seq_len
        
        pdb.set_trace()
        packed_stories = pack_padded_sequence(img_padded_sentence, lens, enforce_sorted=False)
        # TODO: bug here!
        pdb.set_trace()
        output, hidden = self.rnn(packed_stories, hidden)
        return output, hidden


class ModelV1(nn.Module):
    def __init__(self, vocab):
        super(ModelV1, self).__init__()
        self.encoder = Encoder()
        self.decoders = [Decoder(vocab_size=len(vocab)) for i in range(NUM_SENTS)]

        self.vocab = vocab
        self.out_layer = nn.Linear(HIDDEN_SIZE, len(vocab))
        self.vocab_length = len(vocab)
        self.logSoftmax = nn.LogSoftmax(dim=2)

    def get_decoded_output(self, decoder_input, hidden, lens):
        # TODO: adapt this
        output, hidden = self.decoder(decoder_input, hidden, lens)
        output, _ = pad_packed_sequence(output)
        output = self.out_layer(output)
        return output, hidden

    def forward(self, images, stories, story_lens):
        """
        :param images: input images (batch_size * num_pic * 3 * width * height)
        :param stories: padded input story sentences (num_sent * batch_size * max_sentence_len)
        :param story_lens: input story sentences lengths (num_sent * batch_size)
        :return:
        """
        num_sent, batch_size, max_sent_len = stories.shape

        embedded, _, hidden = self.encoder(images)
        # embedded: num_pic * batch_size * embedding_dim
        # hidden: 1 * batch_size * hidden_size

        out_story = torch.zeros((num_sent, max_sent_len, batch_size, HIDDEN_SIZE))
        out_story_lens = torch.zeros((num_sent, batch_size))

        for i in range(NUM_SENTS):
            image_embed_i = embedded[i, :, :]
            story_i = stories[i, :, :]
            story_len_i = story_lens[i, :]
            out_i, _ = self.decoders[i](image_embed_i, story_i, hidden, story_len_i)
            out_i, out_lens = pad_packed_sequence(out_i)
            # out_i: ((max_seq_len+1) * batch_size * hidden_size)
            out_story[i] = out_i[1:, ]  # don't want the word predicted by the image embedding
            out_story_lens[i] = (out_lens - 1)
        
        pdb.set_trace()
        ###############################################
        ## TODO: adapt everything below this point!!###
        ###############################################
        n_tokens = story_lens.sum() - story_lens.size(0)

        loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction='sum')

        out, out_lens = pad_packed_sequence(out)
        out = self.out_layer(out)
        out = self.logSoftmax(out)

        for i in range(out.size()[0] - 1):
            active = i + 1 < story_lens
            loss += criterion(out[i, active, :], stories[i + 1, active])

        loss /= n_tokens

        return loss, out
