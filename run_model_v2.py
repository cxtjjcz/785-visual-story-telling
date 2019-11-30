import sys, pdb, os, time
import os.path as osp

import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from modules import LSTM, Attention, VariationalDropout, global_weight_init
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence, pack_padded_sequence
from hyperparams import *

"""
Changes:
    (1) Added Visual Attention for each Decoder
    (2) For implementation purposes, Decoder only has 1 layer :-)
          
TODOs:
    (1) Check our decoders
    (2) Check our loss
    (3) Add input and output drop
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
        self.num_pixles = FEATURE_MAP_DIM * FEATURE_MAP_DIM
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
        feature_vec = torch.zeros((num_pics, batch_size, self.encoder_dim)).to(DEVICE)
        feature_map = torch.zeros((num_pics, batch_size, self.num_pixles, self.encoder_dim)).to(
            DEVICE)

        for i in range(num_pics):
            batch_i = images[:, -(i + 1), :, :, :]  # ith pics
            feature_map_i, feature_vec_i = self.fc7(batch_i)
            feature_map[i] = feature_map_i.view(batch_size, -1, self.encoder_dim)
            feature_vec[i] = feature_vec_i

        output, hidden = self.rnn(feature_vec, hidden)
        # feature_vec: (num_pic, batch_size, encoder_dim)
        # feature_map: (num_pic, batch_size, feature_map_dim**2, encoder_dim)
        # output: (num_pic, batch_size, hidden_size)
        # hidden: if LSTM: tuple of (h_n, c_n): (num_layers, num_directions, batch_size, hidden_size) * 2
        return feature_map, output, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim):
        super(Decoder, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.encoder_dim = encoder_dim
        self.num_pixels = FEATURE_MAP_DIM * FEATURE_MAP_DIM
        self.embedding_size = EMBEDDING_SIZE
        self.attention_dim = ATTENTION_DIM
        self.vocab_size = vocab_size
        self.num_layers = NUM_LAYERS_DECODER

        self.embedding = nn.Embedding(vocab_size, self.embedding_size, padding_idx=3).to(DEVICE)
        self.attention = Attention(encoder_dim=self.encoder_dim,
                                   decoder_dim=self.hidden_size,
                                   attention_dim=self.attention_dim)

        self.f_beta = nn.Linear(self.hidden_size, self.encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.input_drop = VariationalDropout(INPUT_DROPOUT, batch_first=True)
        self.output_drop = VariationalDropout(OUTPUT_DROPOUT, batch_first=True)

        # TODO: change to LSTMCell! (Multiple Layers?)

        self.decode_step = nn.LSTMCell(self.embedding_size + self.encoder_dim,
                                       self.hidden_size, bias=True)
        # Note: LSTMCell is BATCH FIRST!!!!!

    def init_hidden_state(self, encoder_hidden):
        """
        https://github.com/bastings/annotated_encoder_decoder/blob/master/annotated_encoder_decoder.ipynb
        :param encoder_hidden: encoder's final hidden state (h_n, c_n for LSTM)
        :return: decoder_initial_hidden 2 * (batch_size, decoder_hidden)
        """
        h_n_s, c_n_s = encoder_hidden  # each has shape (num_layers * num_directions, batch_size, encoder_hidden)

        fwd_hidden = h_n_s[-2]
        bwd_hidden = h_n_s[-1]
        final_hidden = torch.cat([fwd_hidden, bwd_hidden], dim=1)
        # (batch_size, decoder_hidden)

        fwd_cell = c_n_s[-2]
        bwd_cell = c_n_s[-1]
        final_cell = torch.cat([fwd_cell, bwd_cell], dim=1)
        # (batch_size, decoder_hidden)

        return final_hidden, final_cell

    def forward(self, image_embedding, padded_sentence, encoder_hidden, sentence_lens):
        """
        :param image_embedding: image embedding for the corresponding sentence:
                                (batch_size, feature_map_dim**2, encoder_dim)
        :param padded_sentence: (batch_size * max_seq_len)
        :param encoder_hidden: encoder's final hidden state
        :param sentence_lens: (batch_size)
        :return:
        """
        batch_size = image_embedding.size(0)
        padded_sentence = self.embedding(padded_sentence)  # (batch_size, max_seq_len, embedding_size)
        h, c = self.init_hidden_state(encoder_hidden)  # each is (batch_size, hidden_dim)
        ordered_sentence_lens, ordered_idx = sentence_lens.sort(descending=True)
        image_embedding = image_embedding[ordered_idx]
        padded_sentence = padded_sentence[ordered_idx]
        h, c = h[ordered_idx], c[ordered_idx]

        # input dropout
        # TODO: fix bug here
        # padded_sentence = self.input_drop(padded_sentence)

        decode_lengths = sentence_lens.tolist()
        max_seq_len = max(decode_lengths)

        alphas = torch.zeros(batch_size, max_seq_len, self.num_pixels).to(DEVICE)
        outputs = torch.zeros(batch_size, max_seq_len, self.hidden_size).to(DEVICE)

        for t in range(max_seq_len):
            # TODO: check correctness
            batch_size_t = sum([l > t for l in decode_lengths])
            h_active, c_active = h[0:batch_size_t], c[0:batch_size_t]
            context, alpha = self.attention(image_embedding[0:batch_size_t],
                                            h_active)
            gate = self.sigmoid(self.f_beta(h_active))  # gating scalar, (batch_size, encoder_dim)
            gated_context = gate * context
            h, c = self.decode_step(torch.cat([padded_sentence[0:batch_size_t, t, :],
                                               gated_context], dim=1),
                                    (h_active, c_active))
            outputs[:batch_size_t, t, :] = h
            alphas[:batch_size_t, t] = alpha

        # put them back
        outputs = outputs[ordered_idx]
        alphas = alphas[ordered_idx]

        # output dropout
        # TODO: fix bug here
        # outputs = self.output_drop(outputs)
        return outputs, alphas


class ModelV2(nn.Module):
    def __init__(self, vocab):
        super(ModelV2, self).__init__()
        self.encoder = Encoder()
        self.decoders = nn.ModuleList([Decoder(vocab_size=len(vocab),
                                               encoder_dim=self.encoder.encoder_dim)
                                       for i in range(NUM_SENTS)])
        self.attention_pixel_dim = self.decoders[0].num_pixels
        self.vocab = vocab
        self.out_layer = nn.Linear(HIDDEN_SIZE, len(vocab))
        self.vocab_length = len(vocab)
        self.logSoftmax = nn.LogSoftmax(dim=2)
        self.criterion = nn.NLLLoss(reduction='sum')
        super().apply(global_weight_init)

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
        # embedded: (num_pic, batch_size, feature_map_dim**2, encoder_dim)
        # hidden: ????

        out_story = torch.zeros((num_sent, batch_size, max_sent_len, HIDDEN_SIZE)).to(DEVICE)
        out_attention = torch.zeros((num_sent, batch_size, max_sent_len, self.attention_pixel_dim)).to(DEVICE)
        out_story_lens = story_lens.clone()  # story_len does not change


        for i in range(NUM_SENTS):
            image_embed_i = embedded[i, :, :]
            story_i = stories[i, :, :]
            story_len_i = story_lens[i, :]
            # NOTE: inside decoder, we pack_padded_sequence the ith sentences and then pad_packed_sequence.
            # However, the max_seq_len changes to the maximum value for this batch of sentences
            # instead of the global max_seq_len for all sentences
            out_i, alpha_i = self.decoders[i](image_embed_i, story_i, hidden, story_len_i)
            # out_i, out_lens = pad_packed_sequence(out_i)
            # out_i: (batch_size, max_seq_len_batch * hidden_size)
            end_length = out_i.size(1)
            out_story[i, :, 0:end_length, :] = out_i
            out_attention[i, :, 0:end_length, :] = alpha_i

        # TODO: check loss computation
        n_tokens = 0
        loss = 0.0
        out_probs = []

        for i in range(NUM_SENTS):
            out_i = self.out_layer(out_story[i, :, :, :])
            score_i = self.logSoftmax(out_i)
            out_probs.append(score_i.cpu())
            story_len_i = out_story_lens[i, :]
            ground_truth_story_i = stories[i, :, :]
            for j in range(score_i.size(1) - 1):
                active = j + 1 < story_len_i
                if active.sum() == 0:
                    break
                n_tokens += active.sum()
                loss += self.criterion(score_i[active, j, :], ground_truth_story_i[active, j + 1])

        loss /= n_tokens

        return loss, out_probs, out_attention

