import sys, pdb, os, time
import os.path as osp
from hyperparams import *
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from beam_search import *

def train(epochs, model, train_dataloader, optimizer):
    model.train()
    model.to(DEVICE)
    for epoch in range(epochs):
        print('=' * 20)
        print('Epoch: ', epoch)
        print('=' * 20)
        avg_loss = 0
        total_avg_loss = 0
        start_time = time.time()
        for batch_num, (images, sents, sents_len) in enumerate(train_dataloader):
            print(images.shape, sents.shape, sents_len.shape)
            optimizer.zero_grad()
            print(batch_num)
            # Process data and put on device
            images = images.float()
            sents = sents.long()
            sents_len = sents_len.long()
            images, sents, sents_len = images.to(DEVICE), sents.to(DEVICE), sents_len.to(DEVICE)

            # Run through model
            loss, output = model(images, sents, sents_len)

            avg_loss += loss.item()
            total_avg_loss += loss.item()

            loss.backward()

            clipping_value = 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

            optimizer.step()

        model_path = 'Training/'
        torch.save(model.state_dict(), model_path + str(epoch))
        print('Total Loss: ', total_avg_loss / len(train_dataloader))
        
        end_time = time.time()
        print('Total Epoch Time: ', end_time - start_time)


def get_unprocessed_sent(data, sents, vocab):
    references = []
    hypotheses = []
    # def process(vec):
    #     return vocab.vec2sent(vec[np.where(vec>3.0)])
    # references = np.apply_along_axis(process, 1, data)
    # hypotheses = np.apply_along_axis(process, 1, sents)
    for i in range(data.shape[1]):
        ref_vec = (sents[:, i])
        hyp_vec = (data[:, i])
        sentence = vocab.vec2sent(hyp_vec[np.where(hyp_vec > 3.0)])
        real_sentence = vocab.vec2sent(ref_vec[np.where(ref_vec > 3.0)])
        references.append(real_sentence)
        hypotheses.append(sentence)
    return references, hypotheses


def get_bleu(refs, hyps, mode="write", ref_path="refs", hyp_path="hyps"):
    if mode == "write":
        with open(ref_path, "w") as f:
            f.write("\n".join(refs))
            f.close()
        with open(hyp_path, "w") as f:
            f.write("\n".join(hyps))
            f.close()
    elif mode == "read":
        refs = open(ref_path, "r").read().splitlines()
        hyps = open(hyp_path, "r").read().splitlines()
    smoothing_f = SmoothingFunction().method4  # baseline
    refs = [[ref.split()] for ref in refs]
    hyps = [hyp.split() for hyp in hyps]
    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothing_f)
    print("BLEU Score: %.4f" % bleu)


def test(model, dataloader, vocab):
    # Place model in test mode
    with torch.no_grad():
        model.eval()
        model.to(DEVICE)
        start_time = time.time()  # Timeit
        references = []
        hypotheses = []
        for batch_num, (images, sents, sents_len) in enumerate(dataloader):
            # Process data and put on device
            images = images.float()
            sents = sents.long()
            sents_len = sents_len.long()
            images, sents, sents_len = images.to(DEVICE), sents.to(DEVICE), sents_len.to(DEVICE)
            # Run through model
            loss, output = model(images, sents, sents_len)
            decoder_output = torch.argmax(output, dim=2)
            decoder_input = decoder_output.view(-1, sents.size()[
                1])  # (-1, batch_size) fixed batch_size at 1 for testing for now
            decoder_input = decoder_input.cpu().detach().numpy()
            sents = sents.cpu().detach().numpy()
            refs, hyps = get_unprocessed_sent(decoder_input, sents, vocab)
            references.extend(refs)
            hypotheses.extend(hyps)

        get_bleu(references, hypotheses, mode="write")
        end_time = time.time()  # Timeit
        print('Total Test Time: ', end_time - start_time)


def test_v2(model, dataloader, vocab):
    # greedy
    with torch.no_grad():
        model.eval()
        model.to(DEVICE)
        start_time = time.time()
        references = []
        hypotheses = []
        for batch_num, (images, sents, sents_len) in enumerate(dataloader):
            print(images.shape, sents.shape, sents_len.shape)
            print(batch_num)
            # Process data and put on device
            images = images.float()
            sents = sents.long()
            sents_len = sents_len.long()
            images, sents, sents_len = images.to(DEVICE), sents.to(DEVICE), sents_len.to(DEVICE)
            hypothesis = greedy_decode_v2(model, images, vocab)
            reference = get_reference(sents, vocab)
            print("Hyp: " + hypothesis)
            print("Ref: " + reference)
            hypotheses.append(hypothesis)
            references.append(reference)
        get_bleu(references, hypotheses, mode="write")
        end_time = time.time()  # Timeit
        print('Total Test Time: ', end_time - start_time)


def test_v2_tf(model, dataloader, vocab):
    with torch.no_grad():
        model.eval()
        model.to(DEVICE)
        for batch_num, (images, sents, sents_len) in enumerate(dataloader):
            loss, out_probs = model(images, sents, sents_len)
            for i in range(NUM_SENTS):
                out_prob = out_probs[i]
                out = torch.argmax(out_prob, dim=1)
                print(out)