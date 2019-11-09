import sys, pdb, os, time
import os.path as osp
from hyperparams import *
import torch

def train(epochs, model, train_dataloader, optimizer, device):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        print('='*20)
        print('Epoch: ', epoch)
        print('='*20)
        avg_loss = 0
        total_avg_loss = 0
        start_time = time.time()
        start_batch_time = time.time()
        for batch_num, (images, sents, sents_len) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Process data and put on device
            images = images.float()
            sents = sents.long()
            sents_len = sents_len.long()
            images, sents, sents_len = images.to(device), sents.to(device), sents_len.to(device)
            
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
        end_time = time.time()
        
        print('Total Epoch Time: ', end_time - start_time)
        print('Total Loss: ', total_avg_loss / len(train_dataloader))
        
def get_unprocessed_sent(data, sents,  vocab):
    sentence = vocab.vec2sent(data[:, 0])
    real_sentence = vocab.vec2sent(sents[:, 0])
    print('='*20)
    print('SENTENCE: ', sentence)
    print()
    print('REAL: ', real_sentence)
    print()
    print('='*20)
    pdb.set_trace()
    
def test(model, dataloader, device, vocab):
    # Place model in test mode
    with torch.no_grad():
        model.eval()
        model.to(device)
        start_time = time.time() # Timeit
        
        for batch_num, (images, sents, sents_len) in enumerate(dataloader):
            # Process data and put on device
            images = images.float()
            sents = sents.long()
            sents_len = sents_len.long()
            images, sents, sents_len = images.to(device), sents.to(device), sents_len.to(device)

            # Run through model
            loss, output = model(images, sents, sents_len)
            decoder_output = torch.argmax(output, dim=2)
            decoder_input = decoder_output.view(-1, 1) # (-1, batch_size) fixed batch_size at 1 for testing for now
            decoder_input = decoder_input.cpu().detach().numpy()
            sents = sents.cpu().detach().numpy()
            get_unprocessed_sent(decoder_input, sents, vocab)

        end_time = time.time() # Timeit
        print('Total Test Time: ', end_time - start_time)

