import argparse
from collections import defaultdict
import pathlib
import statistics
from typing import List


class Shard:
    def __init__(self, filename) -> None:
        self.filename = filename
        self.articles = []

    def __len__(self) -> int:
        return sum(len(a) for a in self.articles)

    def flush(self) -> None:
        with open(self.filename, mode='w', newline='\n') as f:
            f.write("\n\n".join(str(a) for a in self.articles))


class Article:
    def __init__(self) -> None:
        self.paragraphs = []

    def __len__(self) -> int:
        return sum(len(p) for p in self.paragraphs)

    def __str__(self) -> str:
        return "[PAR]\n".join(str(p) for p in self.paragraphs)


class Paragraph:
    def __init__(self) -> None:
        self.sentences = []

    def __len__(self) -> int:
        return len(self.sentences)

    def __str__(self) -> str:
        return '\n'.join(self.sentences)


class ShardProcessor:
    def __init__(self, input_path: str, output_name_prefix: str, n_train_shards=16, n_valid_shards=16) -> None:
        assert n_train_shards > 0, 'There must be at least one output shard.'
        assert n_valid_shards > 0, 'There must be at least one output shard.'

        self.input_path = input_path

        self.n_train_shards = n_train_shards
        self.n_valid_shards = n_valid_shards

        self.output_name_prefix = output_name_prefix
        pathlib.Path(self.output_name_prefix).mkdir(parents=True, exist_ok=True)
        self.output_train_identifier = 'train'
        self.output_valid_identifier = 'valid'
        self.output_file_extension = '.md'

        self.train_articles = []
        self.valid_articles = []
        self.train_shards = []    # key: filename, value: list of articles to go into file
        self.valid_shards = []        # key: filename, value: list of articles to go into file

        self._init_shards()

    def _init_shards(self) -> None:
        print('Start: Init Output Files', flush=True)
        for i in range(self.n_train_shards):
            filename = f"{self.output_name_prefix}/{self.output_train_identifier}_{i}{self.output_file_extension}"
            self.train_shards.append(Shard(filename))

        for i in range(self.n_valid_shards):
            filename = f"{self.output_name_prefix}/{self.output_valid_identifier}_{i}{self.output_file_extension}"
            self.valid_shards.append(Shard(filename))

        print('End: Init Output Files', flush=True)

    def load_articles(self) -> None:
        print('Start: Load articles', flush=True)

        self.train_articles = self._load_articles(f"{self.input_path}/train.md")
        self.valid_articles = self._load_articles(f"{self.input_path}/valid.md")

        print(f"Number of train articles: {len(self.train_articles)} with {sum(len(a) for a in self.train_articles)} sentences", flush=True)
        print(f"Number of valid articles: {len(self.valid_articles)} with {sum(len(a) for a in self.valid_articles)} sentences", flush=True)
        print('End: Load articles', flush=True)

    def _load_articles(self, input_file) -> List[Article]:
        articles = []
        with open(input_file) as f:
            article, paragraph = Article(), Paragraph()
            for sentence in f.readlines():
                sentence = sentence.strip()

                if (len(sentence) == 0 or sentence.startswith('#')) and len(paragraph) > 0:
                    article.paragraphs.append(paragraph)
                    paragraph = Paragraph()

                if sentence.startswith("# ") and len(article) > 0:
                    articles.append(article)
                    article = Article()

                if len(sentence) > 0:
                    paragraph.sentences.append(sentence)

            if len(paragraph) > 0:
                article.paragraphs.append(paragraph)
            if len(article) > 0:
                articles.append(article)

        return articles

    def distribute_articles_over_shards(self) -> None:
        print('Start: Distribute Articles Over Shards', flush=True)

        self._distribute_articles_over_shards(self.train_articles, self.train_shards)
        self._distribute_articles_over_shards(self.valid_articles, self.valid_shards)

        print("End: Distribute Articles Over Shards", flush=True)

    def _distribute_articles_over_shards(_, articles, shards) -> None:
        assert len(articles) >= len(shards), 'There are fewer articles than shards. Please add more data or reduce the number of shards requested.'

        # Create dictionary with - key: sentence count per article, value: article id number
        sentence_counts = defaultdict(lambda: [])

        max_sentences, total_sentences = 0, 0
        for article_index, article in enumerate(articles):
            current_length = len(article)
            sentence_counts[current_length].append(article_index)
            max_sentences = max(max_sentences, current_length)
            total_sentences += current_length

        n_sentences_per_shard = total_sentences // len(shards)
        assert all(len(article) < n_sentences_per_shard for article in articles)

        consumed_article_set = set()
        unused_article_set = set(range(len(articles)))

        # Make first pass and add one article worth of lines per file
        for shard in shards:
            current_article_id = sentence_counts[max_sentences].pop()
            shard.articles.append(articles[current_article_id])
            consumed_article_set.add(current_article_id)
            unused_article_set.remove(current_article_id)

            # Maintain the max sentence count
            while len(sentence_counts[max_sentences]) == 0 and max_sentences > 0:
                max_sentences -= 1

        counts = [len(shard) for shard in shards]
        median = statistics.median(counts)

        print(f"Median of sentences after the first pass: {median}", flush=True)

        # Make subsequent passes over files to find articles to add without going over limit
        history_remaining = []
        n_history_remaining = 4

        while len(consumed_article_set) < len(articles):
            for shard_index, shard in enumerate(shards):
                # Skip adding to this file, will come back later if no file can accept unused articles
                if counts[shard_index] > median:
                    continue

                # Maintain the max sentence count
                while len(sentence_counts[max_sentences]) == 0 and max_sentences > 0:
                    max_sentences -= 1

                next_article_size = min(n_sentences_per_shard - counts[shard_index], max_sentences)
                while len(sentence_counts[next_article_size]) == 0 and next_article_size > 0:
                    next_article_size -= 1

                # Skip adding to this file, will come back later if no file can accept unused articles
                if next_article_size == 0:
                    continue

                current_article_id = sentence_counts[next_article_size].pop()

                shard.articles.append(articles[current_article_id])
                consumed_article_set.add(current_article_id)
                unused_article_set.remove(current_article_id)

            # If unable to place articles a few times, bump up nominal sizes by fraction until articles get placed
            if len(history_remaining) == n_history_remaining:
                history_remaining.pop(0)
            history_remaining.append(len(unused_article_set))

            history_same = True
            for i in range(1, len(history_remaining)):
                history_same = history_same and (history_remaining[i-1] == history_remaining[i])

            if history_same:
                n_sentences_per_shard += 1

            counts = [len(shard) for shard in shards]
            median = statistics.median(counts)

            print(f"Distributing data over shards: {len(unused_article_set)} articles remaining.")

        if len(unused_article_set) != 0:
            print("Warning: Some articles did not make it into output files.")

        for shard in shards:
            print(f"shard {shard.filename}: {len(shard)} sentences")

    def write_shards_to_disk(self) -> None:
        print('Start: Write Shards to Disk', flush=True)
        for shard in self.train_shards + self.valid_shards:
            shard.flush()
        print('End: Write Shards to Disk', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT sharding')
    parser.add_argument('--input_path', type=str, default="data/pretrain/bnc", help='Specify the input files prefix')
    parser.add_argument('--output_path', type=str, default="data/pretrain/shards", help='Specify the output files prefix')
    parser.add_argument('--n_train_shards', type=int, default=4, help='Specify the number of train shards to generate')
    parser.add_argument('--n_valid_shards', type=int, default=1, help='Specify the number of valid shards to generate')
    args = parser.parse_args()

    processor = ShardProcessor(args.input_path, args.output_path, args.n_train_shards, args.n_valid_shards)
    processor.load_articles()
    processor.distribute_articles_over_shards()
    processor.write_shards_to_disk()
