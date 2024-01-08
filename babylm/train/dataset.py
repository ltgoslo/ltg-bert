import torch
from torch.utils.data import Dataset
from smart_open import open


class SpanMaskingStrategy:
    def __init__(self, mask_p, tokenizer, n_special_tokens, padding_label_id=-100, random_p=0.1, keep_p=0.1):
        self.mask_p = mask_p
        self.random_p = random_p
        self.keep_p = keep_p
        self.tokenizer = tokenizer
        self.n_special_tokens = n_special_tokens
        self.padding_label_id = padding_label_id
        self.mask_index = self.tokenizer.token_to_id("[MASK]")

    def __call__(self, tokens):
        labels = torch.full_like(tokens, fill_value=self.padding_label_id)
        inputs = tokens.clone()

        n_masked = torch.binomial((tokens >= self.n_special_tokens).float().sum(dim=0, keepdim=True), torch.FloatTensor([self.mask_p])).item()
        n_masked = min((tokens >= self.n_special_tokens).long().sum(dim=0), max(1, n_masked))
        preservation_mask = tokens < self.n_special_tokens
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        counter = 100

        while n_masked > mask.long().sum() and counter > 0:
            span_length = torch.tensor([0]).geometric_(1/3).item() % 10
            offset = torch.randint(-(span_length - 1), tokens.size(0) + span_length, []).item()
            sub_mask = torch.zeros_like(tokens, dtype=torch.bool)
            sub_mask[max(0, offset) : min(mask.size(0)-1, offset + span_length)] = True
            sub_mask[preservation_mask] = False

            random_p = torch.rand([]).item()

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            if random_p < 1.0 - self.random_p - self.keep_p:
                inputs[sub_mask] = self.mask_index
            elif random_p < 1.0 - self.keep_p:
                random_words = torch.randint(
                    low=self.n_special_tokens - 1,
                    high=self.tokenizer.get_vocab_size(),
                    size=(sub_mask.sum(),),
                    dtype=torch.long
                )
                inputs[sub_mask] = random_words
            else:
                inputs[sub_mask] = tokens[sub_mask]

            mask |= sub_mask
            counter -= 1

        labels[mask] = tokens[mask]

        return inputs, labels


class Dataset(Dataset):
    def __init__(self, file, offset: int, n_gpus: int, tokenizer, seq_length=512, mask_p=0.15, short_p=0.1, random_p=0.1, keep_p=0.1):
        self.tokenizer = tokenizer

        self.seq_length = seq_length
        self.short_p = short_p
        self.n_special_tokens = 6

        self.masking_strategy = SpanMaskingStrategy(mask_p, tokenizer, self.n_special_tokens, padding_label_id=-100, random_p=random_p, keep_p=keep_p)

        self.mask_index = self.tokenizer.token_to_id("[MASK]")
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.pad_index = self.tokenizer.token_to_id("[PAD]")

        self.segments = []
        for i, segment in enumerate(open(file, "r")):
            if i % n_gpus != offset:
                continue

            segment = segment.strip().split(" ")
            assert len(segment) <= seq_length - 2, " ".join(segment)
            segment = [self.tokenizer.token_to_id(token) for token in segment]
            self.segments.append(segment)

    def __len__(self):
        return len(self.segments)
    
    def rand(self):
        return torch.rand(1).item()

    def randint(self, low, high):
        return torch.randint(low=low, high=high, size=(1,)).item()

    def __getitem__(self, index):
        tokens = self.segments[index]

        target_seq_length = self.seq_length - 2 if self.rand() > self.short_p else self.randint(1, self.seq_length - 2)
        tokens = tokens[:target_seq_length]
        padding_length = (self.seq_length - 2) - len(tokens)
        segment = [self.cls_index] + tokens + [self.sep_index] + [self.pad_index] * padding_length
        segment = torch.LongTensor(segment)

        attention_mask = torch.cat([
            torch.zeros(len(tokens) + 2, dtype=torch.bool),
            torch.ones(padding_length, dtype=torch.bool)
        ])

        inputs, outputs = self.masking_strategy(segment)

        return inputs, attention_mask, outputs

    def show_random_item(self):
        inputs, _, outputs = self.__getitem__(self.randint(0, len(self)))
        print(' '.join(self.tokenizer.id_to_token(i) for i in inputs), flush=True)
        print(' '.join(self.tokenizer.id_to_token(o) if o >= 0 else "-1" for o in outputs), flush=True)
