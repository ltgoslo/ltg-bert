import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean, stdev
import math

from tqdm import tqdm
import wandb
import torchmetrics
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tokenizers import Tokenizer

from glue_dataset import GlueDataset, GlueCollateFunctor, to
from config import BertConfig
from deberta import Bert, FeedForward
from deberta_diff_classifier import Bert as BertDiffClassifier
from deberta_embedding_ln import Bert as BertEmbeddingLN
from deberta_no_embedding_ln import Bert as BertNoEmbeddingLN
from utils import seed_everything
from lazy_adam import LazyAdamW


TASK_LABELS = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "stsb": 1,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "ax": 3,
    "cb": 3,
    "wic": 2,
    "copa": 1,
}


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_dir", default=None, type=str, help="Path to GLUE.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="The initial checkpoint to start training from.")
    parser.add_argument("--task", default="cola", type=str, help="GLUE task.")
    parser.add_argument("--lr", default=3.0e-5, type=float, help="BERT learning rate.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")

    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--vocab_path", default="../bert-lumi/data/pretrain/wordpiece_vocab.json", type=str, help="The vocabulary the BERT model will train on.")

    args = parser.parse_args()

    return args


def setup_training(args):
    assert torch.cuda.is_available()

    seed_everything(args.seed)

    device = torch.device("cuda")
    args.n_gpu = 1

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

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

    return config, model


class CopaMetric:
    def update(self, predictions, targets, ids):
        for p, t, i in zip(predictions.tolist(), targets.tolist(), ids.tolist()):
            if i not in self.predictions:
                self.predictions[i] = p
                self.targets[i] = t
            else:
                self.correct += 1 if (self.predictions[i] > p) == (self.targets[i] > t) else 0
                self.total += 1

    def reset(self):
        self.predictions = {}
        self.targets = {}
        self.correct, self.total = 0, 0

    def compute(self):
        return torch.tensor(self.correct / self.total)


if __name__ == "__main__":
    args = parse_arguments()

    device, args, checkpoint = setup_training(args)
    if wandb.run is None:
        wandb.init(
            name=checkpoint["args"].run_name,
            id=checkpoint["args"].wandb_id,
            project="bert-bnc",
            entity="ltg",
            resume="auto",
            mode="offline",
            allow_val_change=True,
        )

    tokenizer = Tokenizer.from_file(args.vocab_path)
    config, bert = prepare_model(checkpoint, device)

    train_set = GlueDataset(f"{args.input_dir}/{args.task}_train.pickle", args.task, 512)
    if args.task != "mnli":
        valid_set = {
            "validation": GlueDataset(f"{args.input_dir}/{args.task}_validation.pickle", args.task)
        }
    else:
        valid_set = {
            "validation_matched": GlueDataset(f"{args.input_dir}/{args.task}_validation_matched.pickle", args.task),
            "validation_mismatched": GlueDataset(f"{args.input_dir}/{args.task}_validation_mismatched.pickle", args.task),
            "validation_hans": GlueDataset(f"{args.input_dir}/hans_validation.pickle", "hans")
        }

    if args.task == "cola":
        metrics = {
            "MCC": torchmetrics.MatthewsCorrCoef(num_classes=TASK_LABELS[args.task])
        }
    elif args.task in ["mrpc", "qqp"]:
        metrics = {
            "accuracy": torchmetrics.Accuracy(),
            "f1": torchmetrics.F1Score(num_classes=TASK_LABELS[args.task], multiclass=False)
        }
    elif args.task in ["cb"]:
        metrics = {
            "accuracy": torchmetrics.Accuracy(),
            "f1": torchmetrics.F1Score(num_classes=TASK_LABELS[args.task], multiclass=True, average="macro")
        }
    elif args.task == "stsb":
        metrics = {
            "pearson_correlation": torchmetrics.PearsonCorrCoef(),
            "spearman_correlation": torchmetrics.SpearmanCorrCoef()
        }
    elif args.task == "copa":
        metrics = {
            "accuracy": CopaMetric(),
        }
    else:
        metrics = {
            "accuracy": torchmetrics.Accuracy()
        }

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=GlueCollateFunctor(tokenizer, 512),
        num_workers=4,
        pin_memory=True
    )
    valid_loader = {
        key: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=GlueCollateFunctor(tokenizer, 512),
            num_workers=4,
            pin_memory=True
        )
        for key, dataset in valid_set.items()
    }

    class Classifier(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
            self.hidden = nn.Linear(config.hidden_size, config.hidden_size)
#            self.mlp = FeedForward(config)
            self.dropout = nn.Dropout(0.1)
            self.output = nn.Linear(config.hidden_size, TASK_LABELS[args.task])

            self.initialize(config.hidden_size)

        def initialize(self, hidden_size):
            std = math.sqrt(2.0 / (5.0 * hidden_size))
            nn.init.trunc_normal_(self.hidden.weight, mean=0.0, std=std, a=-2*std, b=2*std)
            nn.init.trunc_normal_(self.output.weight, mean=0.0, std=std, a=-2*std, b=2*std)
            self.hidden.bias.data.zero_()
            self.output.bias.data.zero_()

        def forward(self, x):
#            x = self.dropout(x)
#            x = x + self.mlp(x)
#            x = self.layer_norm(x)
            x = self.dropout(x)
            x = F.relu(self.hidden(x))
            x = self.layer_norm(x)
            x = self.dropout(x)
            return self.output(x)

    classifier = Classifier(config).to(device)

    if args.task == "stsb":
        classifier.output.bias.data.fill_(2.5)

    no_decay = ['bias', "layer_norm", "_embedding"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in bert.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": args.lr
        },
        {
            "params": [p for n, p in bert.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.lr
        },
        {
            "params": [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": 1.0e-3
        },
        {
            "params": [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": 1.0e-3
        }
    ]

    optimizer = LazyAdamW(optimizer_grouped_parameters, eps=1e-6)
    scheduler = lr_scheduler.ChainedScheduler([
        lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1.0, total_iters=args.epochs*len(train_loader) * 0.01),
        lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1e-9, total_iters=args.epochs*len(train_loader))
    ])

    if args.task == "stsb":
        criterion = nn.MSELoss()
    elif args.task == "copa":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    results_history = []

    for epoch in range(args.epochs):
        bert.train()
        classifier.train()
        for batch in tqdm(train_loader):
            batch = to(batch, device)
            optimizer.zero_grad(set_to_none=True)

            pooled_output = bert.get_contextualized(batch["input_ids"].t(), batch["attention_mask"])[-1][0, :, :]
            predictions = classifier(pooled_output).squeeze(-1)

            loss = criterion(predictions, batch["labels"].float() if args.task == "copa" else batch["labels"])
            loss.backward()
#            grad_norm = nn.utils.clip_grad_norm_(bert.parameters(), 1.0, norm_type=2.0)
#            print(grad_norm.item())
            optimizer.step()
            scheduler.step()

        bert.eval()
        classifier.eval()
        with torch.no_grad():
            results = {}
            for split, loader in valid_loader.items():
                for metric in metrics.values():
                    metric.reset()
                for batch in loader:
                    batch = to(batch, device)
                    pooled_output = bert.get_contextualized(batch["input_ids"].t(), batch["attention_mask"])[0, :, :]
                    predictions = classifier(pooled_output).squeeze(-1)
                    if split == "validation_hans":
                        predictions = torch.softmax(predictions, dim=-1)
                        predictions = torch.stack([
                            torch.sum(predictions[:, :2], dim=-1),
                            predictions[:, -1]
                        ], dim=1)

                    for metric in metrics.values():
                        metric.update(
                            predictions.cpu(),
                            batch["labels"].cpu(),
                            *([batch["ids"].cpu()] if args.task == "copa" else [])
                        )

                for metric_name, metric in metrics.items():
                    results[f"{split}_{metric_name}"] = metric.compute().item() * 100.0
                    print(f"$$$ {epoch}\t{metric.compute().item() * 100.0}", flush=True)

            results_history.append(results)
            print(results, flush=True)

    for key, value in results.items():
        wandb.run.summary[f"{args.task}/{key}"] = value

    results_history = results_history[-len(results_history) // 2:]
    results_history_ = defaultdict(list)
    for results in results_history:
        for key, value in results.items():
            results_history_[key].append(value)

    stats = {}
    for key, values in results_history_.items():
        stats[f"glue/max/{args.task}_{key}"] = max(values)
        stats[f"glue/min/{args.task}_{key}"] = min(values)
        stats[f"glue/mean/{args.task}_{key}"] = mean(values)
        stats[f"glue/std/{args.task}_{key}"] = stdev(values)

    print(stats, flush=True)


    # if args.task == "mnli":
    #     checkpoint_path = f"{'/'.join(args.checkpoint_path.split('/')[:-1])}/mnli.bin"
    #     torch.save(bert.state_dict(), checkpoint_path)
