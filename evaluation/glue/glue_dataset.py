import pickle
import gzip
from itertools import chain

import torch
from torch.utils.data import Dataset


def to(obj, device):
    if isinstance(obj, dict):
        return {
            key: to(value, device)
            for key, value in obj.items()
        }

    if torch.is_tensor(obj):
        return obj.to(device)
    else:
        return obj


class GlueDataset(Dataset):
    def __init__(self, input_path, task, max_length=512):
        self.max_length = max_length

        with gzip.open(input_path, mode='rb') as f:
            self.examples = pickle.load(f)

        for example in self.examples:
            for key, value in example.items():
                if key in ["sentences", "span_masks"]:
                    example[key] = self.truncate_sentences(value)
                elif key == "entity":
                    example[key] = value
                else:
                    example[key] = torch.tensor(value)

    def truncate_sentences(self, sentences):
        max_length = self.max_length - 1 - len(sentences)  # without special tokens
        sentences = [torch.tensor(sentence) for sentence in sentences]
        sentence_lengths = [sentence.size(0) for sentence in sentences] + [0]
        sorted_indices = [i for i, _ in sorted(enumerate(sentence_lengths), key=lambda x: -x[1])]
        total_length = sum(sentence_lengths)
        i = 0
        while total_length > max_length:
            diff = sentence_lengths[sorted_indices[i]] - sentence_lengths[sorted_indices[i + 1]]
            for j in range(i + 1):
                to_remove = min(diff, (total_length - max_length + j) // (i + 1))
                j = sorted_indices[j]
                sentences[j] = sentences[j][:sentence_lengths[j] - to_remove]
                sentence_lengths[j] -= to_remove
            total_length = sum(sentence_lengths)
            i += 1

        return sentences

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


class GlueCollateFunctor:
    def __init__(self, tokenizer, max_length):
        self.cls_index = torch.LongTensor([tokenizer.token_to_id("[CLS]")])
        self.sep_index = torch.LongTensor([tokenizer.token_to_id("[SEP]")])
        self.pad_index = tokenizer.token_to_id("[PAD]")
        self.max_length = max_length

    def __call__(self, samples):
        subword_lengths = [
            sum(sentence.size(0) + 1 for sentence in sample["sentences"]) + 1
            for sample in samples
        ]
        for i in range(len(samples)):
            if subword_lengths[i] > self.max_length:
                samples[i]["sentences"][0] = samples[i]["sentences"][0][:-(subword_lengths[i] - self.max_length)]
                subword_lengths[i] = self.max_length

        max_subword_length = max(subword_lengths)

        output = {
            "input_ids": torch.stack([
                self.merge_tokens(sample["sentences"], subword_lengths[i], max_subword_length)
                for i, sample in enumerate(samples)
            ]).long(),
            "segment_ids": torch.stack([
                self.merge_segment_ids(sample["sentences"], subword_lengths[i], max_subword_length)
                for i, sample in enumerate(samples)
            ]),
            "attention_mask": torch.stack([
                torch.cat([
                    torch.zeros(subword_lengths[i], dtype=torch.bool),
                    torch.ones(max_subword_length - subword_lengths[i], dtype=torch.bool)
                ], dim=-1)
                for i, _ in enumerate(samples)
            ]),
            "labels": torch.stack([sample["label"] for sample in samples]),
            "ids": torch.stack([sample["id"] for sample in samples]),
        }

        if "span_masks" in samples[0]:
            output["span_masks"] = torch.stack([
                self.merge_span_masks(sample["span_masks"], subword_lengths[i], max_subword_length)
                for i, sample in enumerate(samples)
            ])

        if "entity" in samples[0]:
            output["entity"] = [sample["entity"] for sample in samples]

        return output

    def merge_tokens(self, sentences, length, total_length):
        tensors = [
            self.cls_index,
            *chain(*((sentence, self.sep_index) for sentence in sentences)),
            torch.full((total_length - length,), fill_value=self.pad_index, dtype=torch.long)
        ]
        return torch.cat(tensors)

    def merge_span_masks(self, sentences, length, total_length):
        tensors = [
            torch.ones(1, dtype=torch.bool),
            *chain(*((sentence, torch.ones(1, dtype=torch.bool)) for sentence in sentences)),
            torch.full((total_length - length,), fill_value=1, dtype=torch.bool)
        ]
        return torch.cat(tensors)

    def merge_segment_ids(self, sentences, length, total_length):
        tensors = [
            torch.zeros(1, dtype=torch.long),
            *(torch.full((sentence.size(0) + 1,), fill_value=i % 2, dtype=torch.long) for i, sentence in enumerate(sentences)),
            torch.zeros(total_length - length, dtype=torch.long)
        ]
        return torch.cat(tensors)


def truncate_sentences(sentences, max_length):
    max_length = max_length - 1 - len(sentences)  # without special tokens
    sentences = [torch.tensor(sentence) for sentence in sentences]
    sentence_lengths = [sentence.size(0) for sentence in sentences] + [0]
    sorted_indices = [i for i, _ in sorted(enumerate(sentence_lengths), key=lambda x: -x[1])]
    total_length = sum(sentence_lengths)
    i = 0
    while total_length > max_length:
        diff = sentence_lengths[sorted_indices[i]] - sentence_lengths[sorted_indices[i + 1]]
        for j in range(i + 1):
            to_remove = min(diff, (total_length - max_length + j) // (i + 1))
            j = sorted_indices[j]
            sentences[j] = sentences[j][:sentence_lengths[j] - to_remove]
            sentence_lengths[j] -= to_remove
        total_length = sum(sentence_lengths)
        i += 1

    return sentences
