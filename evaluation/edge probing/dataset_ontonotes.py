import pickle
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def safe_cat(tensors, dim=0):
    if len(tensors) == 0:
        return torch.FloatTensor([])
    else:
        return torch.cat(tensors, dim=dim)


def safe_stack(tensors, dim=0):
    if len(tensors) == 0:
        return torch.FloatTensor([])
    else:
        return torch.stack(tensors, dim=dim)


def to(obj, device):
    if isinstance(obj, dict):
        return {
            key: to(value, device)
            for key, value in obj.items()
        }

    assert torch.is_tensor(obj)
    return obj.to(device)


class EdgeProbingDataset(Dataset):
    def __init__(self, path, vocab, max_length=126):
        with gzip.open(path, mode='rb') as f:
            self.sentences = pickle.load(f)
        self.sentences = [s for s in self.sentences if len(s["subwords"]) > 0 and len(s["subwords"]) <= max_length]
        self.vocab = vocab

    def __getitem__(self, index):
        sentence = self.sentences[index]

        output = {
            "input_ids": torch.LongTensor(sentence["subwords"]),
            "const": {
                "labels": safe_stack([
                    F.one_hot(torch.LongTensor(const["label"]), num_classes=len(self.vocab["const"])).sum(0)
                    for const in sentence["const"]
                ]),
                "length": len(sentence["const"]),
                "span_mask_1": self.create_span_mask(sentence, "const", "span_1")
            },
            "coref": {
                "labels": safe_stack([
                    F.one_hot(torch.LongTensor(coref["label"]), num_classes=len(self.vocab["coref"])).sum(0)
                    for coref in sentence["coref"]
                ]),
                "length": len(sentence["coref"]),
                "span_mask_1": self.create_span_mask(sentence, "coref", "span_1"),
                "span_mask_2": self.create_span_mask(sentence, "coref", "span_2")
            },
            "entities": {
                "labels": safe_stack([
                    F.one_hot(torch.LongTensor(entities["label"]), num_classes=len(self.vocab["entities"])).sum(0)
                    for entities in sentence["entities"]
                ]),
                "length": len(sentence["entities"]),
                "span_mask_1": self.create_span_mask(sentence, "entities", "span_1")
            },
            "pos": {
                "labels": safe_stack([
                    F.one_hot(torch.LongTensor(pos["label"]), num_classes=len(self.vocab["pos"])).sum(0)
                    for pos in sentence["pos"]
                ]),
                "length": len(sentence["pos"]),
                "span_mask_1": self.create_span_mask(sentence, "pos", "span_1")
            },
            "srl": {
                "labels": safe_stack([
                    F.one_hot(torch.LongTensor(srl["label"]), num_classes=len(self.vocab["srl"])).sum(0)
                    for srl in sentence["srl"]
                ]),
                "length": len(sentence["srl"]),
                "span_mask_1": self.create_span_mask(sentence, "srl", "span_1"),
                "span_mask_2": self.create_span_mask(sentence, "srl", "span_2")
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


class EdgeProbingCollateFunctor:
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
                torch.ones(max_subword_length - subword_lengths[i], dtype=torch.bool)
            ], dim=-1)
            for i, _ in enumerate(samples)
        ])

        for task in ["const", "coref", "entities", "pos", "srl"]:
            output[task] = {}
            output[task]["labels"] = safe_cat([
                sample[task]["labels"]
                for sample in samples
            ], dim=0)

            for span_type in (["span_mask_1", "span_mask_2"] if (task in ["coref", "srl"]) else ["span_mask_1"]):
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
