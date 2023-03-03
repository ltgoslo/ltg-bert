import pickle
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset_ontonotes import *


class UdDataset(Dataset):
    def __init__(self, path, vocab, max_length=126):
        with gzip.open(path, mode='rb') as f:
            self.sentences = pickle.load(f)
        self.sentences = [s for s in self.sentences if len(s["subwords"]) > 0 and len(s["subwords"]) <= max_length]
        self.vocab = vocab

    def __getitem__(self, index):
        sentence = self.sentences[index]
        output = {
            "input_ids": torch.LongTensor(sentence["subwords"]),
            "ud": {
                "labels": safe_stack([
                    F.one_hot(torch.LongTensor(span_label["label"]), num_classes=len(self.vocab)).sum(0)
                    for span_label in sentence["ud"]
                ]),
                "length": len(sentence["ud"]),
                "span_mask_1": self.create_span_mask(sentence, "ud", "span_1"),
                "span_mask_2": self.create_span_mask(sentence, "ud", "span_2")
            },
        }
        return output

    def __len__(self):
        return len(self.sentences)

    def create_span_mask(self, sentence, task: str, span_type: str):
        return torch.BoolTensor(
            [
                [
                    False if i >= pos[span_type][0] and i <= pos[span_type][1] else True
                    for i in range(len(sentence["subwords"]))
                ]
                for pos in sentence[task]
            ]
        )


class UdCollateFunctor:
    def __init__(self, tokenizer):
        self.cls_index = torch.LongTensor([tokenizer.token_to_id("[CLS]")])
        self.sep_index = torch.LongTensor([tokenizer.token_to_id("[SEP]")])
        self.pad_index = tokenizer.token_to_id("[PAD]")

    def __call__(self, samples):
        subword_lengths = [sample["input_ids"].size(0) + 2 for sample in samples]
        max_subword_length = max(subword_lengths)

        output = {}
        output["input_ids"] = torch.stack([
            torch.cat([
                self.cls_index,
                sample["input_ids"],
                self.sep_index,
                torch.full([max_subword_length - subword_lengths[i]], fill_value=self.pad_index, dtype=torch.long)
            ])
            for i, sample in enumerate(samples)
        ])
        output["segment_ids"] = torch.zeros(len(samples), max_subword_length, dtype=torch.long)
        output["attention_mask"] = torch.stack([
            torch.cat([
                torch.zeros(subword_lengths[i], dtype=torch.bool),
                torch.ones( max_subword_length - subword_lengths[i], dtype=torch.bool)
            ], dim=-1)
            for i, _ in enumerate(samples)
        ])

        for task in ["ud"]:
            output[task] = {}
            output[task]["labels"] = safe_cat([
                sample[task]["labels"]
                for sample in samples
            ], dim=0)

            for span_type in ["span_mask_1", "span_mask_2"]:
                output[task][span_type] = safe_cat([
                    F.pad(sample[task][span_type], pad=(1, 1 + max_subword_length - subword_lengths[i]), value=True)
                    for i, sample in enumerate(samples)
                    if sample[task]["labels"].size(0) > 0
                ], dim=0)

            output[task]["sentence_indices"] = safe_cat([
                torch.full([sample[task]["length"]], fill_value=i, dtype=torch.long)
                for i, sample in enumerate(samples)
            ])

        return output
