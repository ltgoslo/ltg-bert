import argparse
import gzip
import pickle
import os

os.environ['HF_DATASETS_OFFLINE'] = "1" 

from tokenizers import Tokenizer
from datasets import load_dataset
from collections import Counter
import torch

from subtokenize_tokenized import normalize, detokenize


def tokenize_sentence(sentence, task, tokenizer):
    sentence = sentence.strip()
    sentence = normalize(sentence)

    if task in ["sst2", "mrpc", "wsc", "wsc.fixed", "hans"]:
        sentence = detokenize(sentence.split(' '))[0]

    return tokenizer.encode(sentence, add_special_tokens=False).ids


def process_split(dataset, task, tokenizer):
    examples, lengths = [], Counter()
    for i, example in enumerate(dataset):
        if task in ["cola", "sst2"]:
            examples.append({
                "id": example["idx"],
                "sentences": [
                    tokenize_sentence(example["sentence"], task, tokenizer)
                ],
                "label": example["label"]
            })

        elif task in ["mrpc", "stsb", "rte", "wnli"]:
            examples.append({
                "id": example["idx"],
                "sentences": [
                    tokenize_sentence(example["sentence1"], task, tokenizer),
                    tokenize_sentence(example["sentence2"], task, tokenizer)
                ],
                "label": example["label"]
            })

        elif task in ["mnli", "hans"]:
            examples.append({
                "id": example["idx"] if "idx" in example else i,
                "sentences": [
                    tokenize_sentence(example["premise"], task, tokenizer),
                    tokenize_sentence(example["hypothesis"], task, tokenizer)
                ],
                "label": example["label"]
            })

        elif task in ["qqp"]:
            examples.append({
                "id": example["idx"],
                "sentences": [
                    tokenize_sentence(example["question1"], task, tokenizer),
                    tokenize_sentence(example["question2"], task, tokenizer)
                ],
                "label": example["label"]
            })

        elif task in ["qnli"]:
            examples.append({
                "id": example["idx"],
                "sentences": [
                    tokenize_sentence(example["question"], task, tokenizer),
                    tokenize_sentence(example["sentence"], task, tokenizer)
                ],
                "label": example["label"]
            })

        elif task in ["boolq"]:
            examples.append({
                "id": example["idx"],
                "sentences": [
                    tokenize_sentence(example["question"], task, tokenizer),
                    tokenize_sentence(example["passage"], task, tokenizer)
                ],
                "label": example["label"]
            })
        elif task in ["cb"]:
            examples.append({
                "id": example["idx"],
                "sentences": [
                    tokenize_sentence(example["premise"], task, tokenizer),
                    tokenize_sentence(example["hypothesis"], task, tokenizer)
                ],
                "label": example["label"]
            })
        elif task in ["copa"]:
            if example["question"] == "cause":
                examples.append({
                    "id": example["idx"],
                    "sentences": [
                        tokenize_sentence(example["choice1"], task, tokenizer),
                        tokenize_sentence(example["premise"], task, tokenizer),
                    ],
                    "label": 1 - example["label"]
                })
                examples.append({
                    "id": example["idx"],
                    "sentences": [
                        tokenize_sentence(example["choice2"], task, tokenizer),
                        tokenize_sentence(example["premise"], task, tokenizer),
                    ],
                    "label": example["label"]
                })
            else:
                examples.append({
                    "id": example["idx"],
                    "sentences": [
                        tokenize_sentence(example["premise"], task, tokenizer),
                        tokenize_sentence(example["choice1"], task, tokenizer),
                    ],
                    "label": 1 - example["label"]
                })
                examples.append({
                    "id": example["idx"],
                    "sentences": [
                        tokenize_sentence(example["premise"], task, tokenizer),
                        tokenize_sentence(example["choice2"], task, tokenizer),
                    ],
                    "label": example["label"]
                })
        elif task in ["multirc"]:
            examples.append({
                "id": example["idx"]["question"],
                "sentences": [
                    tokenize_sentence(example["paragraph"], task, tokenizer),
                    tokenize_sentence(example["question"], task, tokenizer),
                    tokenize_sentence(example["answer"], task, tokenizer)
                ],
                "label": example["label"]
            })
        elif task in ["record"]:
            for entity in example["entities"]:
                examples.append({
                    "id": example["idx"]["query"],
                    "sentences": [
                        tokenize_sentence(example["passage"], task, tokenizer),
                        tokenize_sentence(example["query"], task, tokenizer),
                        tokenize_sentence(entity, task, tokenizer)
                    ],
                    "label": 1 if entity in example["answers"] else 0,
                    "entity": entity
                })

        elif task in ["wic"]:
            sentence_1_encoding = tokenizer.encode(example["sentence1"], add_special_tokens=False)
            sentence_2_encoding = tokenizer.encode(example["sentence2"], add_special_tokens=False)

            sentence_1_mask = [offset[0] < example["start1"] or offset[1] > example["end1"] for offset in sentence_1_encoding.offsets]
            sentence_2_mask = [offset[0] < example["start2"] or offset[1] > example["end2"] for offset in sentence_2_encoding.offsets]

            if not all(sentence_1_mask):
                sentence_1_mask = [False for _ in sentence_1_mask]
            if not all(sentence_2_mask):
                sentence_2_mask = [False for _ in sentence_2_mask]

            examples.append({
                "id": example["idx"],
                "sentences": [
                    sentence_1_encoding.ids,
                    sentence_2_encoding.ids
                ],
                "span_masks": [
                    sentence_1_mask,
                    sentence_2_mask
                ],
                "label": example["label"]
            })
        elif task in ["wsc", "wsc.fixed"]:
            examples.append({
                "id": example["idx"],
                "sentences": [
                    tokenize_sentence(example["text"], task, tokenizer),
                    tokenize_sentence(example["span1_text"], task, tokenizer),
                    tokenize_sentence(example["span2_text"], task, tokenizer)
                ],
                "label": example["label"]
            })

    lengths = Counter([
        2 + sum(len(sentence) + 1 for sentence in example["sentences"])
        for example in examples
    ])
    return examples, lengths


def log_sentence_lengths(output_path: str, task: str, split: str, length_freqs: Counter):
    lengths_cumsum = torch.zeros(max(length_freqs) + 1, dtype=torch.long)
    for length, freq in length_freqs.items():
        lengths_cumsum[length] = freq
    lengths_cumsum = torch.flip(torch.cumsum(torch.flip(lengths_cumsum, [0]), dim=0), [0])

    with open(f"{output_path}/freqs_{task}_{split}.txt", "w") as f:
        for i in range(lengths_cumsum.size(0)):
            f.write(f"{i}\t{lengths_cumsum[i]}\t{lengths_cumsum[i] / lengths_cumsum[0] * 100:.3f}%\n")


def process(output_path: str, dataset_name: str, task: str, tokenizer):
    dataset = load_dataset(dataset_name, task, ignore_verifications=True)

    for key, split in dataset.items():
        data, length_freqs = process_split(split, task, tokenizer)
        print(dataset_name, task, key, "# inputs: ", sum(length_freqs.values()), flush=True)
        log_sentence_lengths(output_path, task, key, length_freqs)

        with gzip.open(f"{output_path}/{task}_{key}.pickle", mode='wb') as f:
            pickle.dump(data, f)


def process_hans(output_path: str, tokenizer):
    dataset = load_dataset("hans")

    for key, split in dataset.items():
        data, length_freqs = process_split(split, "hans", tokenizer)
        print("HANS", key, "# inputs: ", sum(length_freqs.values()), flush=True)
        log_sentence_lengths(output_path, "hans", key, length_freqs)

        with gzip.open(f"{output_path}/hans_{key}.pickle", mode='wb') as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", default="../data/pretrain/wordpiece_vocab.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--output_path", default="../data/extrinsic/glue", type=str, help="The output file where the model checkpoints will be written.")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    process(args.output_path, "glue", "cola", tokenizer)
    process(args.output_path, "glue", "sst2", tokenizer)
    process(args.output_path, "glue", "mrpc", tokenizer)
    process(args.output_path, "glue", "qqp", tokenizer)
    process(args.output_path, "glue", "stsb", tokenizer)
    process(args.output_path, "glue", "mnli", tokenizer)
    process(args.output_path, "glue", "qnli", tokenizer)
    process(args.output_path, "glue", "rte", tokenizer)
    # process(args.output_path, "glue", "wnli", tokenizer)

    process(args.output_path, "super_glue", "boolq", tokenizer)
    process(args.output_path, "super_glue", "cb", tokenizer)
    process(args.output_path, "super_glue", "copa", tokenizer)
    process(args.output_path, "super_glue", "multirc", tokenizer)
    process(args.output_path, "super_glue", "record", tokenizer)
    process(args.output_path, "super_glue", "wic", tokenizer)
    # process(args.output_path, "super_glue", "wsc", tokenizer)
    # process(args.output_path, "super_glue", "wsc.fixed", tokenizer)

    process_hans(args.output_path, tokenizer)
