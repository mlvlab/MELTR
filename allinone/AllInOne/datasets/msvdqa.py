import numpy as np
from .video_base_dataset import BaseDataset
import os
import pandas as pd
import json


class MSVDQADataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.ans_lab_dict = None
        if split == "train":
            names = ["msvd_qa_train"]
        elif split == "val":
            names = ["msvd_qa_test"]  
        elif split == "test":
            names = ["msvd_qa_test"] 

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )
        self._load_metadata()

    def _load_metadata(self):
        metadata_dir = '../../ICLR2023/VideoQA//meta_data/msvd'
        split_files = {'train': 'msvd_train.jsonl', 'val': 'msvd_val.jsonl', 'test': 'msvd_test.jsonl'}
        # split_files = {'train': 'what_train.jsonl', 'val': 'what_test.jsonl', 'test': 'what_test.jsonl'}

        self.ans_lab_dict = {}
        answer_fp = os.path.join(metadata_dir, 'msvd_answer_set.txt')
        self.youtube_mapping_dict = dict()
        with open(os.path.join(metadata_dir, 'msvd_youtube_mapping.txt')) as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split(' ')
                self.youtube_mapping_dict[info[1]] = info[0]
        with open(answer_fp, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                self.ans_lab_dict[str(line.strip())] = count
                count += 1

        split = self.names[0].split('_')[-1]
        target_split_fp = split_files[split]
        self.metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        # for i, k in enumerate(list(self.ans_lab_dict.keys())):
        #     if i == 250:
        #         break
        #     self.metadata = self.metadata[self.metadata['answer'] != k]


        print("total {} samples for {}".format(len(self.metadata), self.names))

    def _get_video_path(self, sample):
        rel_video_fp = self.youtube_mapping_dict['vid' + str(sample["video_id"])] + '.avi'
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_text(self, sample):
        text = sample['question']
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return (text, encoding)

    def get_answer_label(self, sample):
        text = sample['answer']
        ans_total_len = len(self.ans_lab_dict) + 1  
        # ans_total_len = len(self.ans_lab_dict)
        try:
            ans_label = self.ans_lab_dict[text]  
        except KeyError:
            ans_label = -100
        scores = np.zeros(ans_total_len).astype(int)
        scores[ans_label] = 1
        return text, ans_label, scores

    def __getitem__(self, index):
        sample = self.metadata.iloc[index]
        image_tensor = self.get_video(sample)
        text = self.get_text(sample)
        qid = index

        answers, labels, scores = self.get_answer_label(sample)


        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }

    def __len__(self):
        return len(self.metadata) 