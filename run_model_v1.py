import os.path as osp
import time
import pickle
from torch.utils.data import Dataset, DataLoader

from beam_search import *
from model_v1 import ModelV1
from vist_api.vist import Story_in_Sequence
from dataset import StoryDataset, collate_story
from vocab import Vocabulary
from train_test import train, test

vocab_save_path = "vocab.pt"
vist_annotations_dir = '/Users/xiangtic/vist/'
images_dir = '/Users/xiangtic/vist/images/'
sis_train = Story_in_Sequence(images_dir + "toy", vist_annotations_dir)
# sis_val = Story_in_Sequence(images_dir+"val", vist_annotations_dir)
# sis_test = Story_in_Sequence(images_dir+"test", vist_annotations_dir)

cuda = False
# cuda = cuda and torch.cuda.is_available()
device = "cpu"

if (not osp.exists(vocab_save_path)):
    corpus = []
    for story in sis_train.Stories:
        sent_ids = sis_train.Stories[story]['sent_ids']
        for sent_id in sent_ids:
            corpus.append(sis_train.Sents[sent_id]['text'])
    vocab = Vocabulary(corpus, freq_cutoff=1)  # reads and builds

    # Verifying vocabulary is the same
    for word in vocab.w2i.keys():
        index = vocab.w2i[word]
        if (word != vocab.i2w[index]):
            print('Words mismatched...')
    # Saving vocabulary
    with open(vocab_save_path, 'wb') as file:
        pickle.dump(vocab, file)
else:
    vocab = pickle.load(open(vocab_save_path, 'rb'))


def main():
    train_story_set = StoryDataset(sis_train, vocab)
    # val_story_set = StoryDataset(sis_val, vocab)
    # test_story_set = StoryDataset(sis_test, vocab)

    train_loader = DataLoader(train_story_set, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_story,
                              pin_memory=False)
    # imgs of shape [BS, 5, 3, 224, 224]
    # sents BS * 5  * MAX_LEN

    model_v1 = ModelV1(vocab)

    # Learning rate is the most sensitive value to set,
    # will need to test what works well past 400 instances
    optimizer = torch.optim.Adam(model_v1.parameters(), lr=0.001)  # .001 for 400
    isTraining = True

    if isTraining:
        train(10, model_v1, train_loader, optimizer)
    else:
        model_v1.load_state_dict(torch.load('./Training/7'))
        test_loader = DataLoader(train_story_set, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_story)
        test(model_v1, test_loader, device, vocab)


if __name__ == "__main__":
    main()
