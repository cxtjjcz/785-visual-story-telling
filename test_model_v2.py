import pickle
from torch.utils.data import Dataset, DataLoader
from model_v2 import ModelV2
from vist_api.vist import Story_in_Sequence
from dataset import StoryDataset, collate_story
from train_test import *

vocab_save_path = "vocab.pt"
model_ckpt = "40"
vist_annotations_dir = './vist_api'
images_dir = './vist_api/images/'

# minimum for evaluation

vocab = pickle.load(open(vocab_save_path, 'rb'))
print("Vocab size: ", len(vocab))
sis_test = Story_in_Sequence(images_dir+"val", vist_annotations_dir)
test_story_set = StoryDataset(sis_test, vocab)
test_loader = DataLoader(test_story_set, shuffle=False, batch_size=1, collate_fn=collate_story)
model_v2 = ModelV2(vocab)
model_v2.load_state_dict(torch.load(model_ckpt, map_location=DEVICE))

# function call for greedy decoding
# it calls "generate" for each decoder
test_v2(model_v2, test_loader, vocab)

# just to see the intermediate outputs during training
# test_v2_tf(model_v2, test_loader, vocab)