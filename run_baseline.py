from vocab import *
import sys
# sys.path.insert(0, 'vist_api/vist')
from vist_api.vist import *
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import os.path as osp
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
import torch
import pickle
from baseline_model import *
from hyperparams import *
from beam_search import *
import os

vocab_save_path = "vocab.pt"
vist_annotations_dir = '/Users/xiangtic/vist/'
images_dir = '/Users/xiangtic/vist/images/'
sis_train = Story_in_Sequence(images_dir + "train", vist_annotations_dir)
# sis_val = Story_in_Sequence(images_dir+"val", vist_annotations_dir)
# sis_test = Story_in_Sequence(images_dir+"test", vist_annotations_dir)

cuda = True
cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# build/read vocabulary
if not osp.exists(vocab_save_path):
    corpus = []
    for story in sis_train.Stories:
        sent_ids = sis_train.Stories[story]['sent_ids']
        for sent_id in sent_ids:
            corpus.append(sis_train.Sents[sent_id]['text'])
    vocab = Vocabulary(corpus, freq_cutoff=1)
    vocab.build()
    pickle.dump(vocab, open(vocab_save_path, 'wb'))
else:
    vocab = pickle.load(open(vocab_save_path, 'rb'))


# build dataloader
class StoryDataset(Dataset):
    def __init__(self, sis, vocab):
        self.sis = sis
        self.story_indices = list(self.sis.Stories.keys())
        self.vocab = vocab

    def __len__(self):
        return len(self.sis.Stories)

    @staticmethod
    def read_image(path):
        img = Image.open(path)
        img = torchvision.transforms.Resize((224, 224))(img)
        img = torchvision.transforms.ToTensor()(img)
        return img

    def __getitem__(self, idx):
        story_id = self.story_indices[idx]
        story = self.sis.Stories[story_id]
        sent_ids = story['sent_ids']
        img_ids = story['img_ids']
        imgs = []
        for i, img_id in enumerate(img_ids):
            img_file = osp.join(self.sis.images_dir, img_id + '.jpg')
            img_tensor = self.read_image(img_file)
            imgs.append(img_tensor)
        imgs = torch.stack(imgs)

        # container = torch.zeros(MAX_STORY_LEN).fill_(self.vocab["<pad>"])
        sent = ""
        for sent_id in sent_ids:
            sent += self.sis.Sents[sent_id]["text"]
        sent_tensor = self.vocab.sent2vec("<s> " + sent + " </s>")
        # container[:len(sent_tensor)] = sent_tensor
        # sents_packed = pack_sequence(sents)
        return imgs, sent_tensor


def collate_story(seq_list):
    imgs, sents = zip(*seq_list)
    imgs = torch.stack(imgs)
    lens = [len(sent) for sent in sents]
    sents = pad_sequence(sents, padding_value=3)
    return imgs, sents, lens


train_story_set = StoryDataset(sis_train, vocab)
# val_story_set = StoryDataset(sis_val, vocab)
# test_story_set = StoryDataset(sis_test, vocab)


train_loader = DataLoader(train_story_set, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_story)
# imgs of shape [BS, 5, 3, 224, 224]
# sents BS * 5  * MAX_LEN


baseline_model = BaselineModel(vocab)
optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.01)


def train(epochs, model, train_dataloader):
    model.train()
    for epoch in range(epochs):
        avg_loss = 0
        for batch_num, (images, sents, sents_len) in enumerate(train_dataloader):
            optimizer.zero_grad()
            score = model(images, sents, sents_len, device)
            greedy_decode(model, images, device, vocab)
            # comment out to see the current greedy decoded story
            loss = -score
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            if batch_num % PRINT_LOSS == PRINT_LOSS-1:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch + 1, batch_num + 1, avg_loss / 50))
                avg_loss = 0.0

            # torch.cuda.empty_cache()

        # torch.save(model.state_dict(), model_path + "/"+ str(epoch) +".pt")


train(1, baseline_model, train_loader)
