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
    (3) attention!
          
TODOs:

"""


class fc7_Extractor(nn.Module):
    def __init__(self, cnn_type, fine_tune=False):
        super(fc7_Extractor, self).__init__()
        self.cnn_type = cnn_type
        if self.cnn_type == "vgg16":
            # TODO: vgg does not work right now...
            self.feature_dim = 4096
            self.pretrained = models.vgg16(pretrained=True)
            self.fine_tune(fine_tune)
        elif self.cnn_type == "resnet152":
            self.feature_dim = 2048
            model = models.resnet152(pretrained=True)
            modules = list(model.children())[:-2]
            self.pretrained = nn.Sequential(*modules)
            self.fine_tune(fine_tune)
            # pooling layers
            self.pool1 = nn.AdaptiveAvgPool2d((FEATURE_MAP_DIM, FEATURE_MAP_DIM))
            self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if self.cnn_type == "vgg16":
            x = self.pretrained.features(x)
            x = self.pretrained.avgpool(x)
            x = torch.flatten(x, 1)
            x = nn.Sequential(*list(self.pretrained.classifier.children())[:-1])(x)
            return x
        elif "resnet" in self.cnn_type:
            feature_map = self.pool1(self.pretrained(x))  # (batch_size, 2048, encoded_image_size, encoded_image_size)
            feature_vec = self.pool2(feature_map)  # (batch_size, 2048, 1, 1)
            feature_map = feature_map.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
            feature_vec = feature_vec.squeeze()  # (batch_size, 2048)
            return feature_map, feature_vec

    def fine_tune(self, fine_tune):
        if not fine_tune:
            for p in self.pretrained.parameters():
                p.requires_grad = False


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # now, fc7_Extractor returns two things:
        # (1) an intermediate feature map and (2) one final feature vector
        self.fc7 = fc7_Extractor(cnn_type=FEATURE_EXTRACTOR)
        self.encoder_dim = self.fc7.feature_dim
        # divide hidden size by two when using bidirectional rnn
        if BIDIRECTIONAL_ENCODER:
            hidden_size = HIDDEN_SIZE // 2
        else:
            hidden_size = HIDDEN_SIZE

        self.rnn = LSTM(input_size=self.encoder_dim, hidden_size=hidden_size,
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
        feature_map = torch.zeros((num_pics, batch_size, self.encoder_dim)).to(DEVICE)
        feature_vec = torch.zeros((num_pics, batch_size, FEATURE_MAP_DIM * FEATURE_MAP_DIM, self.encoder_dim))

        for i in range(num_pics):
            batch_i = images[:, -(i + 1), :, :, :]  # ith pics
            feature_map_i, feature_vec_i = self.fc7(batch_i)
            feature_map[i, :, :] = feature_map_i
            feature_vec[i, :, :, :] = feature_vec_i.view(batch_size, -1, self.encoder_dim)

        output, hidden = self.rnn(feature_vec, hidden)
        # feature_vec: (num_pic, batch_size, encoder_dim)
        # feature_map: (num_pic, batch_size, feature_map_dim**2, encoder_dim)
        # output: (num_pic, batch_size, hidden_size)
        # hidden: (1, batch_size, hidden_size)
        return feature_map, output, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE, padding_idx=3).to(DEVICE)
        self.rnn = LSTM(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE,
                        bidirectional=BIDIRECTIONAL_DECODER, input_drop=INPUT_DROPOUT,
                        output_drop=OUTPUT_DROPOUT, weight_drop=WEIGHT_DROP,
                        num_layers=NUM_LAYERS_DECODER)

    def forward(self, image_embedding, padded_sentence, hidden, lens):
        """
        :param image_embedding: image embedding for the corresponding sentence: (batch_size * embedding_size)
        :param padded_sentence: (batch_size * max_seq_len)
        :param hidden: encoder's final hidden state
        :param lens: (batch_size * 1)
        :return:
        """
        batch_size = image_embedding.shape[0]
        padded_sentence = self.embedding(padded_sentence)  # (batch_size * max_seq_len * embedding_size)
        # add image embedding to front (as if it's the first word)
        img_padded_sentence = torch.cat((image_embedding.unsqueeze(1), padded_sentence), dim=1)
        # img_padded_sentence : (batch_size * (max_seq_len+1) * embedding_size)
        img_padded_sentence = img_padded_sentence.permute(1, 0, 2)
        # img_padded_sentence : ((max_seq_len+1) * batch_size * embedding_size)
        lens += 1  # add one to max_seq_len

        # output still feels wrong atm
        packed_stories = pack_padded_sequence(img_padded_sentence, lens, enforce_sorted=False)

        # TODO: checkout the Encoder section of
        # https://github.com/bastings/annotated_encoder_decoder/blob/master/annotated_encoder_decoder.ipynb
        fwd_hidden = hidden[0][0:hidden[0].size(0):2]
        bwd_hidden = hidden[0][1:hidden[0].size(0):2]
        final_hidden = torch.cat([fwd_hidden, bwd_hidden], dim=2)  # [num_layers, batch_size, 2*hidden_dim]

        fwd_cell = hidden[1][0:hidden[1].size(0):2]
        bwd_cell = hidden[1][1:hidden[1].size(0):2]
        final_cell = torch.cat([fwd_cell, bwd_cell], dim=2)  # [num_layers, batch_size, 2*hidden_dim]

        output, hidden = self.rnn(packed_stories, (final_hidden, final_cell))
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
        self.criterion = nn.NLLLoss(reduction='sum')

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
        out_story_lens = story_lens.clone()  # story_len does not change

        for i in range(NUM_SENTS):
            image_embed_i = embedded[i, :, :]
            story_i = stories[i, :, :]
            story_len_i = story_lens[i, :]
            # NOTE: inside decoder, we pack_padded_sequence the ith sentences and then pad_packed_sequence.
            # However, the max_seq_len changes to the maximum value for this batch of sentences
            # instead of the global max_seq_len for all sentences
            out_i, _ = self.decoders[i](image_embed_i, story_i, hidden, story_len_i)
            out_i, out_lens = pad_packed_sequence(out_i)
            # out_i: ((max_seq_len_batch+1) * batch_size * hidden_size)
            end_length = out_i[1:, ].shape[0]
            out_story[i, 0:end_length] = out_i[1:, ]  # don't want the word predicted by the image embedding

        # TODO: check loss computation
        n_tokens = 0
        loss = 0.0
        out_probs = []

        for i in range(NUM_SENTS):
            out_i = self.out_layer(out_story[i, :, :, :])
            score_i = self.logSoftmax(out_i)
            out_probs.append(score_i)
            story_len_i = out_story_lens[i, :]
            ground_truth_story_i = stories[i, :, :]
            for j in range(score_i.size()[0] - 1):
                active = j + 1 < story_len_i
                if active.sum() == 0:
                    break
                n_tokens += active.sum()
                loss += self.criterion(score_i[j, active, :], ground_truth_story_i[active, j + 1])

        loss /= n_tokens

        return loss, out_probs
