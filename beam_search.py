# https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
import operator
import torch
from hyperparams import *
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
import torch.nn as nn
import torch.nn.functional as F
from asyncio import PriorityQueue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(target_tensor, decoder_hiddens, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


def get_unprocessed_sent(data, vocab):
    data = data.numpy()
    for batch in range(data.shape[0]):
        vec = data[batch, :]
        if vocab.vec2sent(vec) != '<unk>':
            print(vocab.vec2sent(vec))


def greedy_decode(model, encoder_input, device, vocab):
    """
    :param vocab: vocab class
    :param device: device
    :param encoder_input: images input
    :param model: model
    """
    model.eval()
    batch_size, _, _, _, _ = encoder_input.size()  # batch * 5 * 3 * w * h
    init_hidden = torch.rand(1, batch_size, HIDDEN_SIZE, device=device)
    decoded_batch = torch.zeros((batch_size, MAX_STORY_LEN), device=device)
    decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=device).view(-1, batch_size)
    decoded_batch[:, 0] = decoder_input.view(-1)
    lens = torch.Tensor([1 for i in range(batch_size)])

    _, hidden = model.encoder(encoder_input, init_hidden)

    for t in range(1, MAX_STORY_LEN):
        decoder_output, hidden = model.get_decoded_output(decoder_input, hidden, lens)
        decoder_output = decoder_output.squeeze()
        decoder_output = torch.argmax(decoder_output, dim=1)
        decoded_batch[:, t] = decoder_output
        decoder_input = (decoder_output.view(-1, batch_size))

    model.train()
    get_unprocessed_sent(decoded_batch, vocab)


def get_reference(sents, vocab):
    num_sent, batch_size, max_sent_len = sents.shape
    reference = []

    for i in range(NUM_SENTS):
        sent = sents[i].squeeze()
        # print(sent)
        for word_idx in sent[1:]:
            word_idx = int(word_idx.item())
            reference.append(vocab.i2w[word_idx])
    return " ".join(reference)

def greedy_decode_v2(model, images, vocab):
    """
    :param vocab: vocab class
    :param device: device
    :param encoder_input: images input
    :param model: model
    """
    model.eval()
    embedded, _, hidden = model.encoder(images)
    assert(images.size(0) == 1) # only for for batch_size = 1 now
    hypothesis = []
    for i in range(NUM_SENTS):
        decoder_i = model.decoders[i]
        embedded_i = embedded[i]
        outputs, alphas, words = decoder_i.generate(embedded_i, hidden)
        words = words.squeeze().cpu().numpy()
        for word_idx in words[1:]: # skip SOS
            word_idx = int(word_idx.item())
            if word_idx == vocab["</s>"]: # stop when seeing a EOS
                break
            else:
                hypothesis.append(vocab.i2w[word_idx])
    return " ".join(hypothesis)
