import argparse
from collections import Counter

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers import pre_tokenizers, Regex, normalizers


def initialize_tokenizer(args):
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[PAR]"]

    if args.tokenizer_type.lower() == "bpe":
        model = BPE(unk_token="[UNK]", continuing_subword_prefix='@@')
        trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
            continuing_subword_prefix='@@'
        )
    elif args.tokenizer_type.lower() == "wordpiece":
        model = WordPiece(unk_token="[UNK]")
        trainer = WordPieceTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
            continuing_subword_prefix='@@'
        )
    elif args.tokenizer_type.lower() == "unigram":
        model = Unigram()
        trainer = UnigramTrainer(
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            unk_token="[UNK]"
        )

    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Nmt(),
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.BertNormalizer(
            lowercase=False,  # args.lowercase,
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=True
        )
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Split(
            Regex("▁*(\[UNK\]|\[unk\]|n't|N'T|'ll|'LL|'re|'RE|'ve|'VE|'m|'M|'s|'S|'d|'D)(?!\w+)"),
            behavior="isolated",
            invert=False
        ),
        pre_tokenizers.Split(
            Regex("▁*(\.+|#+|\[UNK\]$|\[unk\]$|n't$|N'T$|'ll$|'LL$|'re$|'RE$|'ve$|'VE$|'m$|'M$|'s$|\
                'S$|'d$|'D$|\d|[^\w▁]){1}"),
            behavior="isolated",
            invert=False
        )
    ])

    return tokenizer, trainer


def calculate_f95(args, tokenizer, f):
    counter = Counter()
    for sentence in f.readlines():
        sentence = sentence.strip()
        if len(sentence) > 0:
            counter.update(tokenizer.encode(sentence).tokens)

    sorted_subwords = counter.most_common()
    print("100 most common subwords:\n" + '\n'.join(str(x) for x in sorted_subwords[:100]) + '\n')

    with open(args.vocab_path + "_freqs", 'w') as f_freq:
        f_freq.write('\n'.join(f"{subword}: {freq}" for subword, freq in sorted_subwords))

    subword95 = sorted_subwords[len(sorted_subwords) * 95 // 100]
    return subword95[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT sharding')
    parser.add_argument('--input_path', type=str, default="data/pretrain/bnc/train.md", help='Specify the input filename')
    parser.add_argument('--vocab_path', type=str, default="data/pretrain/bpe.json", help='Specify the output filename')
    parser.add_argument('--vocab_size', type=int, default=2**14, help='Number of subwords in the trained tokenizer')
    parser.add_argument('--min_frequency', type=int, default=0, help='Minimal number of occurences of every candidate subword')
    parser.add_argument('--tokenizer_type', type=str, default="bpe", help='Type of tokenizer, possible values: {bpe, unigram, wordpiece}')
    parser.add_argument('--lowercase', dest='lowercase', action='store_true', help='Do lowercase everything')
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false', help='Don\'t lowercase anything')
    parser.set_defaults(lowercase=True)
    args = parser.parse_args()

    print(f"Initializing a {args.tokenizer_type} tokenizer", flush=True)
    tokenizer, trainer = initialize_tokenizer(args)

    print("Training the tokenizer", flush=True)
    tokenizer.train([args.input_path], trainer)

    print("Saving the tokenizer", flush=True)
    tokenizer.save(args.vocab_path)

    print("TEST")
    print("Trying to load the tokenizer...")
    tokenizer = Tokenizer.from_file(args.vocab_path)
    print("Success!")

    with open(args.input_path) as f:
        f95 = calculate_f95(args, tokenizer, f)
    print(f"F_{{95%}} is {f95}\n")

    print("Samples from the tokenizer:")

    def test(tokenizer, text):
        subwords = tokenizer.encode(text).tokens
        return ' '.join(subwords)

    texts = [
        """One of the most impressive long term hobby projects is Robert's Rocket Project. He started building a 100 lbf liquid engine in 2001, fired a regeneratively cooled version in 2007, started building a regen 250 lbf in 2008.""",
        """what are examples of interfaces that allow you to manage sets of queries (SQL, splunk, lucene/elastic, xpath, whatever other language)?""",
        """### Increasingly seeing a big schism between what I think my research is & what others think it is. I don't do qualitative work and I'm not trained in anthro or theories of race or gender. I can't supervise students with these interests! I'm a sociophonetician who works on prosody!""",
        """The Northern Lights season is here... Taking these pictures is an art itself and requires preparation, so The Local spoke to an expert to find out how to take awe-inspiring snaps of the Northern Lights.""",
        """Some people have SOTA facial recognition abilities: "At the very upper end of the performance scale, a cohort of just 1-2% of the population are 'super-recognisers'-people who can memorise and recall unfamiliar faces, even after the briefest glimpse.\""""
    ]

    for text in texts:
        print(f"INPUT:  {text}\nTOKENS: {test(tokenizer, text)}\n", flush=True)
