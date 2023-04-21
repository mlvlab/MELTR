from .video_base_dataset import BaseDataset
import random
import os
import pandas as pd
import json
import numpy as np


class MSRVTTDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.cut = "7k"
        if split == "train":
            names = ["msrvtt_train"]
        elif split == "val":
            names = ["msrvtt_val"]
        elif split == "test":
            names = ["msrvtt_val"]
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

        self._load_metadata()

    def _load_metadata(self):
        json_fp = os.path.join(self.data_dir, 'MSR_VTT.json')
        with open(json_fp, 'r') as fid:
            data = json.load(fid)
        df = pd.DataFrame(data['annotations'])

        js_test_cap_idx_path = None
        if self.cut == "7k":
            train_list_path = "train_list_7k.txt"
        elif self.cut == "9k":
            train_list_path = "train_list_9k.txt"
        test_list_path = "val_list_jsfusion.txt" 
        js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"


        train_df = pd.read_csv(os.path.join(self.data_dir, train_list_path), names=['videoid'])
        test_df = pd.read_csv(os.path.join(self.data_dir, test_list_path), names=['videoid'])
        self.split_sizes = {'train': len(train_df), 'val': len(test_df), 'test': len(test_df)}

        if self.split == 'train':
            df = df[df['image_id'].isin(train_df['videoid'])]
        else:
            df = df[df['image_id'].isin(test_df['videoid'])]

        self.metadata = df.groupby(['image_id'])['caption'].apply(list)
        if js_test_cap_idx_path is not None and self.split != 'train':
            caps = pd.Series(np.load(os.path.join(self.data_dir, js_test_cap_idx_path), allow_pickle=True))
            new_res = pd.DataFrame({'caps': self.metadata, 'cap_idx': caps})
            new_res['test_caps'] = new_res.apply(lambda x: [x['caps'][x['cap_idx']]], axis=1)
            self.metadata = new_res['test_caps']

        self.metadata = pd.DataFrame({'captions': self.metadata})
        print("load split {}, {} samples".format(self.split, len(self.metadata)))

    # random choice or fixed?
    def _get_caption(self, sample):
        caption_sample = "rand"
        if self.split in ['train', 'val'] and caption_sample == "rand":
            caption = random.choice(sample['captions'])
        else:
            caption = sample['captions'][0]
        return caption

