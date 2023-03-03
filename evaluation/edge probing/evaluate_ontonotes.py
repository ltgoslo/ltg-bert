import argparse
import torch
import gzip
import pickle
import os

from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from dataset_ontonotes import EdgeProbingCollateFunctor, EdgeProbingDataset, to
from model_edge_probing import EdgeProbingModel
from config import BertConfig
from deberta import Bert
from deberta_diff_classifier import Bert as BertDiffClassifier
from deberta_embedding_ln import Bert as BertEmbeddingLN
from deberta_no_embedding_ln import Bert as BertNoEmbeddingLN
from utils import seed_everything


rank = int(os.environ["SLURM_PROCID"])

CHECKPOINT = [
    "checkpoints/real_subword_mask/model.bin",
    "checkpoints/real_whole_word_mask/model.bin",
    "checkpoints/real_span_mask/model.bin",
    "checkpoints/real_order/model.bin",
    "checkpoints/real_document/model.bin",
    "checkpoints/0.25x_long/model.bin",
    "checkpoints/0.5x_long/model.bin",
    "checkpoints/2x_long/model.bin"
][rank]


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_dir", default=None, type=str, help="Path to ontonotes.")
    parser.add_argument("--config_file", default=None, type=str, help="The BERT model config")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="The initial checkpoint to start training from.")

    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--vocab_path", default="../data/pretrain/wordpiece_vocab.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--n_epochs", default=5, type=int)
    args = parser.parse_args()

    return args


def setup_training(args):
    seed_everything(args.seed)

    device = torch.device("cuda")
    args.n_gpu = 1

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    return device, args, checkpoint


def prepare_model(checkpoint, device):
    config = BertConfig(checkpoint["args"].config_file)

    if not hasattr(checkpoint["args"], "deberta_type") or checkpoint["args"].deberta_type == "normal":
        model = Bert(config, "basic" if not hasattr(checkpoint["args"], "nsp") else checkpoint["args"].nsp)
    elif checkpoint["args"].deberta_type == "diff_classifier":
        model = BertDiffClassifier(config, "basic" if not hasattr(checkpoint["args"], "nsp") else checkpoint["args"].nsp)
    elif checkpoint["args"].deberta_type == "embedding_ln":
        model = BertEmbeddingLN(config, "basic" if not hasattr(checkpoint["args"], "nsp") else checkpoint["args"].nsp)
    elif checkpoint["args"].deberta_type == "no_embedding_ln":
        model = BertNoEmbeddingLN(config, "basic" if not hasattr(checkpoint["args"], "nsp") else checkpoint["args"].nsp)

    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)

    model.requires_grad_(False)

    return config, model


if __name__ == "__main__":
    args = parse_arguments()

    device, args, checkpoint = setup_training(args)

    tokenizer = Tokenizer.from_file(args.vocab_path)
    config, bert = prepare_model(checkpoint, device)

    with gzip.open(f"{args.input_dir}/vocab.pickle", mode='rb') as f:
        edge_probing_vocab = pickle.load(f)
    train_set = EdgeProbingDataset(f"{args.input_dir}/train.pickle", edge_probing_vocab)
    valid_set = EdgeProbingDataset(f"{args.input_dir}/development.pickle", edge_probing_vocab)

    tokenizer = Tokenizer.from_file(args.vocab_path)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, collate_fn=EdgeProbingCollateFunctor(tokenizer))
    valid_loader = DataLoader(valid_set, batch_size=32, collate_fn=EdgeProbingCollateFunctor(tokenizer))

    heads = EdgeProbingModel(config, edge_probing_vocab).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=3.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.n_epochs)

    for _ in range(args.n_epochs):
        i = 0
        bert.train()
        for batch in train_loader:
            batch = to(batch, device)
            optimizer.zero_grad()

            with torch.no_grad():
                layers = bert.get_contextualized(
                    batch["input_ids"].t(),
                    batch["attention_mask"]
                )
                layers = torch.stack(layers[1:]).transpose(1, 2)

            loss = heads(layers, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            if i % 50 == 49:
                print(f"{i}\t" + '    '.join([f"{task} {classifier.metric.compute()*100.0:.2f}%" for task, classifier in heads.heads.items()]), flush=True)
                for classifier in heads.heads.values():
                    classifier.metric.reset()

            i += 1

        for classifier in heads.heads.values():
            classifier.metric.reset()

        bert.eval()
        with torch.no_grad():
            for batch in valid_loader:
                batch = to(batch, device)

                layers = bert.get_contextualized(
                    batch["input_ids"].t(),
                    batch["attention_mask"],
                )
                layers = torch.stack(layers[1:]).transpose(1, 2)
                heads(layers, batch)

        print(f"VALIDATION\t" + '    '.join([f"{task} {classifier.metric.compute()*100.0:.2f}%" for task, classifier in heads.heads.items()]), flush=True)

    for task, classifier in heads.heads.items():
        print(f"$$$ {task} {classifier.metric.compute()*100.0}")

    with open(f"{CHECKPOINT.split('/')[1]}_edge_probing.eval", 'a') as f:
        for task, classifier in heads.heads.items():
            f.write(f"{task} {args.seed} {classifier.metric.compute()*100.0}\n")

    print(f"&&& POS: {torch.nn.functional.softmax(heads.heads['pos'].aggregator.layer_score).tolist()}")
    print(f"&&& SRL: {(0.5 * torch.nn.functional.softmax(heads.heads['srl'].aggregator_1.layer_score) + 0.5 * torch.nn.functional.softmax(heads.heads['srl'].aggregator_2.layer_score)).tolist()}")
    print(f"&&& COREF: {(0.5 * torch.nn.functional.softmax(heads.heads['coref'].aggregator_1.layer_score) + 0.5 * torch.nn.functional.softmax(heads.heads['coref'].aggregator_2.layer_score)).tolist()}")
    print(f"&&& NER: {torch.nn.functional.softmax(heads.heads['entities'].aggregator.layer_score).tolist()}")
