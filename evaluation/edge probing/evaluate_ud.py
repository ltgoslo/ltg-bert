import argparse
import torch
import gzip
import pickle
import os
from tqdm import tqdm

from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from dataset_ud import UdCollateFunctor, UdDataset
from dataset_ontonotes import to
from model_edge_probing import UdModel
from utils import seed_everything
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
    parser.add_argument("--input_dir", default="../data/extrinsic/ud", type=str, help="Path to UD.")
    parser.add_argument("--config_file", default=None, type=str, help="The BERT model config")
    parser.add_argument("--global_step", default=0, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="The initial checkpoint to start training from.")
    parser.add_argument("--n_epochs", default=5, type=int, help="Number of epochs.")

    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--vocab_path", default="../data/pretrain/wordpiece_vocab.json", type=str, help="The vocabulary the BERT model will train on.")

    args = parser.parse_args()

    return args


def setup_training(args):
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        ud_vocab = pickle.load(f)
    train_set = UdDataset(f"{args.input_dir}/en_ewt-ud-train.conllu.pickle", ud_vocab)
    valid_set = UdDataset(f"{args.input_dir}/en_ewt-ud-dev.conllu.pickle", ud_vocab)

    train_loader = DataLoader(train_set, batch_size=128, num_workers=1, shuffle=True, drop_last=True, collate_fn=UdCollateFunctor(tokenizer))
    valid_loader = DataLoader(valid_set, batch_size=128, num_workers=1, collate_fn=UdCollateFunctor(tokenizer))

    classifier = UdModel(config, ud_vocab).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=6.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs * len(train_loader))

    for _ in range(args.n_epochs):
        classifier.head.metric.reset()
        bert.train()
        for i, batch in enumerate(tqdm(train_loader)):
            batch = to(batch, device)
            optimizer.zero_grad()

            with torch.no_grad():
                layers = bert.get_contextualized(
                    batch["input_ids"].t(),
                    batch["attention_mask"]
                )
                layers = torch.stack(layers[1:]).transpose(1, 2)

            loss = classifier(layers, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            if i % 10 == 9:
                print(f"{i}\tud {classifier.head.metric.compute()*100.0:.2f}%")
                classifier.head.metric.reset()

        classifier.head.metric.reset()

    bert.eval()
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            batch = to(batch, device)

            layers = bert.get_contextualized(
                batch["input_ids"].t(),
                batch["attention_mask"]
            )
            layers = torch.stack(layers[1:]).transpose(1, 2)
            classifier(layers, batch)

    print(f"{i}\EVALUATION {classifier.head.metric.compute()*100.0:.2f}%", flush=True)

    print(f"$$$ UD {classifier.head.metric.compute()*100.0}", flush=True)

    with open(f"{CHECKPOINT.split('/')[1]}_edge_probing.eval", 'a') as f:
        f.write(f"UD {args.seed} {classifier.head.metric.compute()*100.0}\n")

    print(f"&&& UD: {(0.5 * torch.nn.functional.softmax(classifier.head.aggregator_1.layer_score) + 0.5 * torch.nn.functional.softmax(classifier.head.aggregator_2.layer_score)).tolist()}")
