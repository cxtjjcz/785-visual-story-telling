import sys, pdb, os, time
import os.path as osp

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
from PIL import Image
import torch.nn.functional as F
from PIL import ImageFile
import torchvision
import numpy as np
import torch
from hyperparams import *


# Build dataset
class StoryDataset(Dataset):
    def __init__(self, sis, vocab):
        self.sis = sis
        self.story_indices = list(self.sis.Stories.keys())
        self.vocab = vocab
        self.numpy_folder = './vist_api/images/Numpys/'
        # self.pre_process()

    def __len__(self):
        #         return 10
        return len(self.story_indices)

    # Check if images have already been processed as numpy arrays, if not save 
    # them as numpy arrays
    def pre_process(self):
        start_time = time.time()
        numpys = set(os.listdir(self.numpy_folder))
        for story_id in self.story_indices:
            story = self.sis.Stories[story_id]
            img_ids = story['img_ids']
            imgs = []
            if (story_id + '.npy' not in numpys):
                for i, img_id in enumerate(img_ids):
                    img_file = osp.join(self.sis.images_dir, img_id + '.jpg')
                    img_tensor = self.read_image(img_file)
                    imgs.append(img_tensor)
                imgs = torch.stack(imgs)
                numpy_name = self.numpy_folder + story_id
                to_save = np.array(imgs)
                np.save(numpy_name, to_save)

        end_time = time.time()
        print('Processing Images Time: ', end_time - start_time)

    @staticmethod
    def read_image(path):
        img = Image.open(path)
        img = torchvision.transforms.Resize((224, 224))(img)
        img = torchvision.transforms.ToTensor()(img)
        # If image is blank and white, make a new tensor and place it inside of it.
        if (img.shape[0] != 3):
            img = img.view(224, 224)
            img = torch.stack([img, img, img])
        return img

    def __getitem__(self, idx):
        story_id = self.story_indices[idx]
        story = self.sis.Stories[story_id]
        my_imgs_path = self.numpy_folder + story_id + '.npy'
        imgs = torch.tensor(np.load(my_imgs_path))

        ## Setence Stuff
        sent_ids = story['sent_ids']
        # sent = ""
        sents = []
        for sent_id in sent_ids:
            # Add a space for the sentence, probably want to just remove punctuation
            # sent += " " + self.sis.Sents[sent_id]["text"]
            sents.append(self.vocab.sent2vec("<s> " + self.sis.Sents[sent_id]["text"] + " </s>"))
        # sent_tensor = self.vocab.sent2vec("<s> " + sent + " </s>")
        # sents_tensor = torch.stack(sents)

        # Return vals
        return imgs, sents


def collate_story(seq_list):
    """

    :param seq_list: [batch images, batch sentences]
    :return: imgs: (batch_size * num_pic * 3 * width * height)
             padded_stories: (num_sent, batch_size, max_seq_len)
             sents_len: (num_sents, batch_size)
    """

    imgs, sents = zip(*seq_list)
    imgs = torch.stack(imgs)
    # sents: a batch (list) of a list of sentences
    sents_len = torch.Tensor([[len(sent) for sent in story] for story in sents])
    batch_max_len = int(sents_len.max().item())

    padded_stories = []
    for story in sents:
        padded_sents = []
        for sent in story:
            if len(sent) < batch_max_len:
                padded_sents.append(F.pad(input=sent,
                                          pad=(0, batch_max_len - len(sent)),
                                          mode="constant", value=3))
            else:
                padded_sents.append(sent)
        padded_stories.append(torch.stack(padded_sents))

    padded_stories = torch.stack(padded_stories).permute(1, 0, 2)
    sents_len = sents_len.permute(1, 0)
    return imgs, padded_stories, sents_len
