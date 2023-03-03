# coding=utf-8

import argparse
import torch
import torch.nn.functional as F
import gzip
import pickle

from tokenizers import Tokenizer

from config import BertConfig
from deberta import Bert

import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_path", default="data/extrinsic/blimp.pickle.gz", type=str, help="Path to BLiMP.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="The initial checkpoint to start training from.")

    # Other parameters
    parser.add_argument("--vocab_path", default="../bert-lumi/data/pretrain/wordpiece_vocab.json", type=str, help="The vocabulary the BERT model will train on.")

    args = parser.parse_args()

    return args


def setup_training(args):
    assert torch.cuda.is_available()

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    return device, args, checkpoint


def prepare_model(checkpoint, device):
    config = BertConfig(checkpoint["args"].config_file)
    model = Bert(config)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)

    return model


def is_right(good, bad, model, tokenizer, device):
    mask_index = tokenizer.token_to_id("[MASK]")
    pad_index = tokenizer.token_to_id("[PAD]")
    cls_index = torch.tensor([tokenizer.token_to_id("[CLS]")], dtype=torch.long)
    sep_index = torch.tensor([tokenizer.token_to_id("[SEP]")], dtype=torch.long)

    good = torch.from_numpy(good).long()
    bad = torch.from_numpy(bad).long()
    labels = torch.cat([good, bad]).unsqueeze(-1).to(device)

    def prepare(tokens, padding: int):
        tokens = torch.cat([cls_index, tokens, sep_index, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - 2 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[1:-(1 + padding), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        attention_mask[:, attention_mask.size(-1) - padding:] = True
        return input_ids, attention_mask

    good_input_ids, good_attention_mask = prepare(good, max(0, len(bad) - len(good)))
    bad_input_ids, bad_attention_mask = prepare(bad, max(0, len(good) - len(bad)))

    logits = model(
        torch.cat([good_input_ids, bad_input_ids], dim=0).t(),
        torch.cat([good_attention_mask, bad_attention_mask], dim=0)
    ).transpose(0, 1)

    indices = torch.cat([torch.arange(1, 1 + len(good), device=device), torch.arange(1, 1 + len(bad), device=device)])
    indices = indices.view(-1, 1, 1).expand(-1, -1, logits.size(-1))
    logits = torch.gather(logits, dim=1, index=indices).squeeze(1)
    log_p = F.log_softmax(logits, dim=-1)

    log_p = log_p.gather(index=labels, dim=-1).squeeze(-1)

    return log_p[:len(good)].sum() > log_p[len(good):].sum()


@torch.no_grad()
def evaluate(model, tokenizer, pairs, device):
    correct = 0
    for pair in pairs:
        good, bad = pair["good"], pair["bad"]

        if is_right(good, bad, model, tokenizer, device):
            correct += 1

    return correct / len(pairs) * 100.0


@torch.no_grad()
def evaluate_all(model, tokenizer, blimp, device):
    total_accuracy, total = 0.0, 0
    for group_key, group in blimp.items():
        total_group_accuracy = 0.0
        for subgroup_key, subgroup in group.items():
            accuracy = evaluate(model, tokenizer, subgroup, device)
            total_group_accuracy += accuracy
            total_accuracy += accuracy
            total += 1

        #     wandb.log(
        #         {
        #             f"blimp_detailed/{group_key}_{subgroup_key}": accuracy
        #         },
        #         step=global_step,
        #         commit=False
        #     )
        #     print(f"{group_key} / {subgroup_key}: {accuracy} %", flush=True)

        # wandb.log(
        #     {
        #         f"blimp/{group_key}": total_group_accuracy / len(group)
        #     },
        #     step=global_step,
        #     commit=False
        # )

    wandb.run.summary["BLiMP/accuracy"] = total_accuracy / total


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
    model = prepare_model(checkpoint, device)
    model.eval()

    with gzip.open(args.input_path, "rb") as f:
        blimp = pickle.load(f)

    evaluate_all(model, tokenizer, blimp, device)
