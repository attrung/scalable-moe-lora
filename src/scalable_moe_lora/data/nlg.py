"""Multi-task dataset loading for LLaMA experiments.

Supports: E2E NLG, CommonGen, WebNLG, SAMSum.
"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CausalLMDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn_dynamic_padding(batch, pad_token_id):
    input_ids_list = [item["input_ids"] for item in batch]
    completion_starts = [item["completion_start"] for item in batch]
    lengths = [len(ids) for ids in input_ids_list]

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)

    attention_mask = torch.zeros_like(input_ids)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1

    labels = input_ids.clone()
    for i, (start, length) in enumerate(zip(completion_starts, lengths)):
        labels[i, :start] = -100
        labels[i, length:] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_collate_fn(pad_token_id):
    def collate(batch):
        return collate_fn_dynamic_padding(batch, pad_token_id)
    return collate


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_e2e(example):
    return example["meaning_representation"], example["human_reference"]

def format_commongen(example):
    return " ".join(example["concepts"]), example["target"]

def format_webnlg(example):
    triples = " | ".join(example["modified_triple_sets"]["mtriple_set"][0])
    return triples, example["lex"]["text"][0]

def format_samsum(example):
    return example["dialogue"], example["summary"]


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_dataset(input_output_pairs, tokenizer, max_seq_len):
    eos_id = tokenizer.eos_token_id
    items = []
    dropped_no_completion = 0
    for context_text, completion_text in input_output_pairs:
        context_ids = tokenizer.encode(context_text)
        completion_ids = tokenizer.encode(" " + completion_text)
        full_ids = context_ids + [eos_id] + completion_ids + [eos_id]
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
        completion_start = len(context_ids) + 1
        # After truncation, if completion_start is past the end of full_ids,
        # all labels in this example become -100 after collate masking. A batch
        # of such examples produces NaN loss (0/0 in CE with ignore_index=-100).
        # Drop these so they can never poison val_loss or gradients.
        if completion_start >= len(full_ids):
            dropped_no_completion += 1
            continue
        items.append({
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "completion_start": completion_start,
        })
    if dropped_no_completion > 0:
        print(f"    [tokenize] dropped {dropped_no_completion} examples "
              f"with context >= max_seq_len (no completion room)")
    return items


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "e2e": {
        "load": lambda split: load_dataset("e2e_nlg", trust_remote_code=True, split=split),
        "format": format_e2e,
        "val_split": "validation",
        "test_split": "test",
    },
    "commongen": {
        "load": lambda split: load_dataset("GEM/common_gen", trust_remote_code=True, split=split),
        "format": format_commongen,
        "val_split": "validation",
        "test_split": "validation",  # test has no references
    },
    "webnlg": {
        "load": lambda split: load_dataset("web_nlg", "release_v3.0_en", trust_remote_code=True, split=split),
        "format": format_webnlg,
        "val_split": "dev",
        "test_split": "test",
    },
    "samsum": {
        "load": lambda split: load_dataset("knkarthick/samsum", trust_remote_code=True, split=split),
        "format": format_samsum,
        "val_split": "validation",
        "test_split": "test",
    },
}


def load_and_prepare_dataset(dataset_name, tokenizer, max_seq_len, split="train"):
    info = DATASET_LOADERS[dataset_name]
    actual_split = info["val_split"] if split == "validation" else split
    ds = info["load"](actual_split)
    fmt = info["format"]

    pairs = []
    for ex in ds:
        try:
            pairs.append(fmt(ex))
        except (IndexError, KeyError):
            continue

    items = tokenize_dataset(pairs, tokenizer, max_seq_len)
    return CausalLMDataset(items)


def load_multitask_dataset(dataset_names, tokenizer, max_seq_len, split="train"):
    all_items = []
    for name in dataset_names:
        ds = load_and_prepare_dataset(name, tokenizer, max_seq_len, split)
        print(f"  {name} ({split}): {len(ds)} examples")
        all_items.extend(ds.items)
    return CausalLMDataset(all_items)


def load_raw_dataset(dataset_name, split="test"):
    info = DATASET_LOADERS[dataset_name]
    actual_split = info["test_split"] if split == "test" else info["val_split"]
    ds = info["load"](actual_split)

    if dataset_name == "e2e":
        from collections import OrderedDict
        ref_map = OrderedDict()
        for ex in ds:
            mr = ex["meaning_representation"]
            if mr not in ref_map:
                ref_map[mr] = []
            ref_map[mr].append(ex["human_reference"])
        return list(ref_map.keys()), list(ref_map.values())

    elif dataset_name == "commongen":
        inputs, references = [], []
        for ex in ds:
            concepts = " ".join(ex["concepts"])
            refs = ex.get("references", [])
            if ex.get("target"):
                refs = [ex["target"]] + [r for r in refs if r != ex["target"]]
            if refs:
                inputs.append(concepts)
                references.append(refs)
        return inputs, references

    elif dataset_name == "webnlg":
        from collections import OrderedDict
        ref_map = OrderedDict()
        for ex in ds:
            try:
                triples = " | ".join(ex["modified_triple_sets"]["mtriple_set"][0])
                texts = [t for t in ex["lex"]["text"] if t.strip()]
                if texts:
                    if triples not in ref_map:
                        ref_map[triples] = []
                    ref_map[triples].extend(texts)
            except (IndexError, KeyError):
                continue
        return list(ref_map.keys()), list(ref_map.values())

    elif dataset_name == "samsum":
        inputs = [ex["dialogue"] for ex in ds]
        references = [[ex["summary"]] for ex in ds]
        return inputs, references

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
