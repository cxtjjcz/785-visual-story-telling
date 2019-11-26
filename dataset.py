import sys, pdb, os, time
import os.path as osp

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
from PIL import Image
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
        self.pre_process()

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
        sent = ""
        for sent_id in sent_ids:
            # Add a space for the sentence, probably want to just remove puncuation
            sent += " " + self.sis.Sents[sent_id]["text"]
        sent_tensor = self.vocab.sent2vec("<s> " + sent + " </s>")

        # Return vals
        return imgs, sent_tensor


def collate_story(seq_list):
    """
    TODO: change sents to be (batch_size, num_sents, max_sentence_length)
          change sents_len to be (batch_size, num_sents, 1)
    """

    imgs, sents = zip(*seq_list)
    imgs = torch.stack(imgs)
    sents_len = torch.Tensor([len(sent) for sent in sents])
    sents = pad_sequence(sents, padding_value=3)
    return imgs, sents, sents_len
