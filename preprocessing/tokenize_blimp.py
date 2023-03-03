# coding=utf-8
import argparse
import numpy as np
import pickle
import gzip
import glob
import json
from collections import defaultdict
from tqdm import tqdm

from tokenizers import Tokenizer


def load(filename):
    pairs = []
    with open(filename) as f:
        for line in f.readlines():
            pairs.append(json.loads(line))
    return pairs


class Processor:
    def __init__(self, input_path, output_path, tokenizer) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.tokenizer = tokenizer

    def load_and_tokenize(self):
        self.documents = [[]]

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        print("creating instance from {}".format(self.input_path))
        total_tokens = 0
        with open(self.input_path, "r") as reader:
            for line in reader.readlines():
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    self.documents.append([])
                    continue

                tokens = self.tokenizer.encode(line, is_pretokenized=False, add_special_tokens=False).ids
                if len(tokens) > 0:
                    self.documents[-1].append(np.array(tokens, dtype=np.int16))
                    total_tokens += len(tokens)

        # Remove empty documents
        self.documents = [document for document in self.documents if len(document) > 0]
        print(f"Loaded {total_tokens} tokens", flush=True)

    def save(self):
        with gzip.open(self.output_path, mode='wb') as f:
            pickle.dump(self.documents, f, protocol=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path", default="../data/pretrain/wordpiece.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--input_path", default="../data/extrinsic/blimp", type=str, help="The input train corpus. can be directory with .txt files or a path to a single file")
    parser.add_argument("--output_path", default="../data/extrinsic/blimp.pickle.gz", type=str, help="The output file where the model checkpoints will be written.")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.vocab_path)

    blimp = defaultdict(lambda: {})
    for dataset in tqdm(glob.glob(f"{args.input_path}/data/*.jsonl")):
        dataset = load(dataset)

        positives = tokenizer.encode_batch(
            [example["sentence_good"] for example in dataset],
            is_pretokenized=False,
            add_special_tokens=False
        )
        negatives = tokenizer.encode_batch(
            [example["sentence_bad"] for example in dataset],
            is_pretokenized=False,
            add_special_tokens=False
        )

        group = dataset[0]["linguistics_term"]
        subgroup = dataset[0]["UID"]
        blimp[group][subgroup] = [
            {
                "good": np.array(positive.ids, dtype=np.int32),
                "bad": np.array(negative.ids, dtype=np.int32)
            }
            for positive, negative
            in zip(positives, negatives)
        ]

        assert len(blimp[group][subgroup]) == 1000

    blimp = dict(blimp)  # we can't pickle defaultdict
    with gzip.open(args.output_path, mode='wb') as f:
        pickle.dump(blimp, f, protocol=-1)
