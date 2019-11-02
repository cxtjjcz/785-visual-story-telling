from vocab import *
import sys
# sys.path.insert(0, 'vist_api/vist')
from vist_api.vist import *
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import os.path as osp
import torch
import pickle
from baseline_model import *
from hyperparams import  *
import os
vocab_save_path = "vocab.pt"
vist_annotations_dir =''
images_dir = '\\e\\images\\'
print(os.listdir('\\e\\images\\train')[:5])
sis_train = Story_in_Sequence(images_dir+"train", vist_annotations_dir)
# sis_val = Story_in_Sequence(images_dir+"val", vist_annotations_dir)
# sis_test = Story_in_Sequence(images_dir+"test", vist_annotations_dir)

cuda =  True
cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# build/read vocabulary
if not osp.exist(vocab_save_path):
    corpus = []
    for story in sis_train.Stories:
        sent_ids = story['sent_ids']
        for sent_id in sent_ids:
            corpus.append(sis_train.Sents[sent_id])
    vocab = Vocabulary(corpus, freq_cutoff=3)
    vocab.build()
    pickle.dump(vocab, open(vocab_save_path, 'wb'))
else:
    vocab = pickle.load(open(vocab_save_path, 'rb'))
print(vocab.w2i("the"))

#
#
# # build dataloader
# class StoryDataset(Dataset):
#     def __init__(self, sis):
#         self.sis = sis
#         self.Stories = sis.Stories
#
#     def __len__(self):
#         return len(self.sis.Stories)
#
#     @staticmethod
#     def read_image(path):
#         img = Image.open(path)
#         img = torchvision.transforms.ToTensor()(img)
#
#     def __getitem__(self, story_id):
#         story = self.Stories[story_id]
#         sent_ids = story['sent_ids']
#         imgs = []
#         for i, sent_id in enumerate(sent_ids):
#             img_id = self.Sents[sent_id]['img_id']
#             img = self.Images[img_id]
#             img_file = osp.join(self.images_dir, img_id + '.jpg')
#             img_tensor = self.read_image(img_file)
#             imgs.append(img_tensor)
#         imgs = torch.stack(imgs)
#
#         sents = []
#         for sent_id in sent_ids:
#             sents.append(self.Sents[sent_id])
#
#         # imgs = [len_story, 3, width, height] numpy array
#         # sents = string list of len_story
#         return imgs, sents
#
# train_story_set = StoryDataset(sis_train, "train")
# val_story_set = StoryDataset(sis_val, "val")
# test_story_set = StoryDataset(sis_test, "test")
#
# baseline_model = BaselineModel(vocab)
# optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.01)
#
#
# def train(epochs, model, train_dataloader):
#     init_hidden = torch.rand(1, 1, model.hidden_size, device=device)
#     model.train()
#     for epoch in epochs:
#         avg_loss = 0
#         for batch_num, (images, sents) in enumerate(train_dataloader):
#             optimizer.zero_grad()
#             loss = -model(images, sents, init_hidden)
#             loss.backward()
#             optimizer.step()
#
#             avg_loss += loss.item()
#
#             if batch_num % 50 == 49:
#                 print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
#                 avg_loss = 0.0
#
#             torch.cuda.empty_cache()
#
#         torch.save(model.state_dict(), path + "/"+ str(epoch) +".pt")
