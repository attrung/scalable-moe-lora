"""Reasoning + NLG dataset loading for LLaMA experiments.

Supports 18 training datasets (v2):
  Reasoning: GSM8K, ARC-Challenge, CommonsenseQA, PIQA, WinoGrande, BoolQ,
             HellaSwag, MATH, OpenBookQA, SciQ, LogiQA2, DROP, MMLU aux_train,
             TriviaQA, ANLI
  Code: MBPP
  NLG: E2E, SAMSum

Eval-only benchmarks (Phase E, loaded via --eval_datasets flag):
  MMLU (test), MMLU-Pro, BBH, IFEval, AGIEval, GPQA Diamond, TruthfulQA
"""

import random

from datasets import load_dataset

from src.data import (
    CausalLMDataset,
    tokenize_dataset,
    format_e2e,
    format_samsum,
)


def format_gsm8k(example):
    return example["question"], example["answer"]


def format_arc(example):
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    choices = " ".join(f"{l}) {t}" for l, t in zip(labels, texts))
    prompt = f"Question: {example['question']}\n{choices}\nAnswer:"
    return prompt, example["answerKey"]


def format_commonsenseqa(example):
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    choices = " ".join(f"{l}) {t}" for l, t in zip(labels, texts))
    prompt = f"Question: {example['question']}\n{choices}\nAnswer:"
    return prompt, example["answerKey"]


def format_piqa(example):
    prompt = f"Goal: {example['goal']}\n1) {example['sol1']}\n2) {example['sol2']}\nAnswer:"
    answer = str(example["label"] + 1)
    return prompt, answer


def format_winogrande(example):
    prompt = f"Sentence: {example['sentence']}\n1) {example['option1']}\n2) {example['option2']}\nAnswer:"
    return prompt, example["answer"]


def format_boolq(example):
    prompt = f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer:"
    answer = "yes" if example["answer"] else "no"
    return prompt, answer


def format_hellaswag(example):
    endings = example["endings"]
    choices = "\n".join(f"{i+1}) {e}" for i, e in enumerate(endings))
    prompt = f"Context: {example['ctx']}\n{choices}\nAnswer:"
    answer = str(int(example["label"]) + 1)
    return prompt, answer


def format_math(example):
    problem = example.get("problem") or example.get("Problem") or ""
    solution = example.get("solution") or example.get("Solution") or ""
    if not problem or not solution:
        return "", ""
    prompt = f"Problem: {problem}\nSolution:"
    return prompt, " " + solution


def format_openbookqa(example):
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    choices = " ".join(f"{l}) {t}" for l, t in zip(labels, texts))
    question = example["question_stem"]
    prompt = f"Question: {question}\n{choices}\nAnswer:"
    return prompt, example["answerKey"]


def format_sciq(example):
    # Deterministic placement by question hash keeps train/eval symmetry.
    question = example["question"]
    correct = example["correct_answer"]
    distractors = [example["distractor1"], example["distractor2"], example["distractor3"]]
    slot = hash(question) % 4
    options = list(distractors)
    options.insert(slot, correct)
    letters = ["A", "B", "C", "D"]
    choices_str = " ".join(f"{l}) {o}" for l, o in zip(letters, options))
    prompt = f"Question: {question}\n{choices_str}\nAnswer:"
    return prompt, letters[slot]


def format_mbpp(example):
    text = example.get("text") or ""
    code = example.get("code") or ""
    tests = example.get("test_list") or []
    if not text or not code:
        return "", ""
    hint = tests[0] if tests else ""
    prompt = f"Problem: {text}\n"
    if hint:
        prompt += f"Example test: {hint}\n"
    prompt += "Solution:\n"
    return prompt, code


def format_logiqa2(example):
    text = example.get("text") or ""
    question = example.get("question") or ""
    options = example.get("options") or []
    answer_idx = example.get("answer")
    if not question or not options or answer_idx is None or len(options) < 4:
        return "", ""
    if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx >= len(options):
        return "", ""
    letters = ["A", "B", "C", "D"]
    choices_str = " ".join(f"{l}) {o}" for l, o in zip(letters, options[:4]))
    prompt = f"Passage: {text}\nQuestion: {question}\n{choices_str}\nAnswer:"
    return prompt, letters[answer_idx]


def format_drop(example):
    passage = example.get("passage") or ""
    question = example.get("question") or ""
    answers = example.get("answers_spans") or {}
    spans = answers.get("spans") or []
    if not passage or not question or not spans:
        return "", ""
    answer = str(spans[0]).strip()
    if not answer:
        return "", ""
    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
    return prompt, answer


def format_mmlu_aux(example):
    # HF loader nests the question under example['train'] in some versions.
    inner = example.get("train") or example
    question = inner.get("question") or ""
    choices = inner.get("choices") or []
    answer_idx = inner.get("answer")
    if not question or len(choices) < 4 or answer_idx is None:
        return "", ""
    if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx >= 4:
        return "", ""
    letters = ["A", "B", "C", "D"]
    choices_str = " ".join(f"{l}) {c}" for l, c in zip(letters, choices[:4]))
    prompt = f"Question: {question}\n{choices_str}\nAnswer:"
    return prompt, letters[answer_idx]


def format_triviaqa(example):
    question = example.get("question") or ""
    answer = example.get("answer") or {}
    value = answer.get("value") or ""
    if not question or not value:
        aliases = answer.get("aliases") or []
        if aliases:
            value = aliases[0]
        else:
            return "", ""
    prompt = f"Question: {question}\nAnswer:"
    return prompt, value.strip()


def format_anli(example):
    # Labels: 0=entailment, 1=neutral, 2=contradiction.
    premise = example.get("premise") or ""
    hypothesis = example.get("hypothesis") or ""
    label = example.get("label")
    if not premise or not hypothesis or label is None:
        return "", ""
    if not isinstance(label, int) or label < 0 or label > 2:
        return "", ""
    letters = ["A", "B", "C"]
    prompt = (
        f"Premise: {premise}\nHypothesis: {hypothesis}\n"
        f"Relationship: A) entailment B) neutral C) contradiction\nAnswer:"
    )
    return prompt, letters[label]


def format_mmlu(example):
    question = example.get("question") or ""
    choices = example.get("choices") or []
    answer_idx = example.get("answer")
    if not question or len(choices) < 4 or answer_idx is None:
        return "", ""
    letters = ["A", "B", "C", "D"]
    choices_str = " ".join(f"{l}) {c}" for l, c in zip(letters, choices[:4]))
    prompt = f"Question: {question}\n{choices_str}\nAnswer:"
    return prompt, letters[answer_idx]


def format_mmlu_pro(example):
    question = example.get("question") or ""
    options = example.get("options") or []
    answer = example.get("answer") or ""
    if not question or not options or not answer:
        return "", ""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(options)]
    choices_str = " ".join(f"{l}) {o}" for l, o in zip(letters, options))
    prompt = f"Question: {question}\n{choices_str}\nAnswer:"
    return prompt, answer.strip().upper()


def format_bbh(example):
    prompt = example.get("input") or ""
    target = example.get("target") or ""
    if not prompt or not target:
        return "", ""
    return prompt, target.strip()


def format_ifeval(example):
    # IFEval has no gold reference; we use a prompt-derived placeholder so
    # BLEU/ROUGE has something to compare against as a cross-variant proxy.
    prompt = example.get("prompt") or ""
    if not prompt:
        return "", ""
    first_sentence = prompt.split(".")[0][:100] if "." in prompt else prompt[:100]
    return prompt, first_sentence


def format_agieval(example):
    question = example.get("question") or ""
    options = example.get("options") or []
    label = example.get("label")
    if not question or not options or label is None:
        return "", ""
    if not isinstance(label, int) or label < 0 or label >= len(options):
        return "", ""
    # Options ship with "(A)", "(B)" prefixes; keep them to match the original benchmark.
    choices_str = "\n".join(options)
    prompt = f"Question: {question}\n{choices_str}\nAnswer:"
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    return prompt, letters[label]


def format_gpqa_diamond(example):
    question = example.get("Question") or example.get("Pre-Revision Question") or ""
    correct = example.get("Correct Answer") or example.get("Pre-Revision Correct Answer") or ""
    incorrect = [
        example.get("Incorrect Answer 1") or example.get("Pre-Revision Incorrect Answer 1"),
        example.get("Incorrect Answer 2") or example.get("Pre-Revision Incorrect Answer 2"),
        example.get("Incorrect Answer 3") or example.get("Pre-Revision Incorrect Answer 3"),
    ]
    incorrect = [x for x in incorrect if x]
    if not question or not correct or len(incorrect) < 3:
        return "", ""
    slot = hash(question) % 4
    options = list(incorrect[:3])
    options.insert(slot, correct)
    letters = ["A", "B", "C", "D"]
    choices_str = " ".join(f"{l}) {o}" for l, o in zip(letters, options))
    prompt = f"Question: {question}\n{choices_str}\nAnswer:"
    return prompt, letters[slot]


def format_truthfulqa(example):
    question = example.get("question") or ""
    mc1 = example.get("mc1_targets") or {}
    choices = mc1.get("choices") or []
    labels = mc1.get("labels") or []
    if not question or not choices or not labels:
        return "", ""
    try:
        correct_idx = labels.index(1)
    except ValueError:
        return "", ""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(choices)]
    if correct_idx >= len(letters):
        return "", ""
    choices_str = "\n".join(f"{l}) {c}" for l, c in zip(letters, choices))
    prompt = f"Question: {question}\n{choices_str}\nAnswer:"
    return prompt, letters[correct_idx]


_BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "formal_fallacies", "geometric_shapes",
    "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]

def _load_bbh(split):
    from datasets import concatenate_datasets
    parts = []
    for task in _BBH_SUBTASKS:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
            parts.append(ds)
        except Exception as e:
            print(f"    [bbh] warning: failed to load {task}: {e}")
    if not parts:
        raise RuntimeError("All BBH subtasks failed to load")
    return concatenate_datasets(parts)


def _load_mbpp(split):
    # MBPP train is only 374 examples; concatenate train+val+test (964) for training.
    # MBPP test is not in the Phase E OOD set, so there is no contamination concern.
    from datasets import concatenate_datasets
    train = load_dataset("google-research-datasets/mbpp", "full", split="train")
    val = load_dataset("google-research-datasets/mbpp", "full", split="validation")
    test = load_dataset("google-research-datasets/mbpp", "full", split="test")
    if split == "train":
        return concatenate_datasets([train, val, test])
    elif split == "validation":
        return val
    else:
        return test


def _load_anli(split):
    # Concatenate train_r1+r2+r3 (~163k); eval uses test_r3 (hardest).
    from datasets import concatenate_datasets
    if split == "train":
        r1 = load_dataset("facebook/anli", split="train_r1")
        r2 = load_dataset("facebook/anli", split="train_r2")
        r3 = load_dataset("facebook/anli", split="train_r3")
        return concatenate_datasets([r1, r2, r3])
    elif split == "validation":
        d1 = load_dataset("facebook/anli", split="dev_r1")
        d2 = load_dataset("facebook/anli", split="dev_r2")
        d3 = load_dataset("facebook/anli", split="dev_r3")
        return concatenate_datasets([d1, d2, d3])
    else:
        return load_dataset("facebook/anli", split="test_r3")


# ~1024 tokens at 4 chars/token — drop examples that would be truncated at seq 1024.
_LONG_MAX_CHARS = 1024 * 4

def _long_passage_filter(prompt, answer):
    return (len(prompt) + len(answer)) <= _LONG_MAX_CHARS


_MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

def _load_math(split):
    from datasets import concatenate_datasets
    parts = []
    for subject in _MATH_SUBJECTS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
            parts.append(ds)
        except Exception as e:
            print(f"    [math] warning: failed to load {subject}/{split}: {e}")
    if not parts:
        raise RuntimeError(f"All MATH subject configs failed for split={split}")
    return concatenate_datasets(parts)


# Drop MATH examples where the solution would be truncated mid-derivation
# at seq 1024, so the model never sees the boxed final answer.
_MATH_MAX_CHARS = 1024 * 4

def _math_train_filter(prompt, answer):
    return (len(prompt) + len(answer)) <= _MATH_MAX_CHARS


REASONING_DATASET_LOADERS = {
    "gsm8k": {
        "load": lambda split: load_dataset("openai/gsm8k", "main", trust_remote_code=True, split=split),
        "format": format_gsm8k,
        "val_split": "test",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "gsm8k",
    },
    "arc": {
        "load": lambda split: load_dataset("allenai/ai2_arc", "ARC-Challenge", trust_remote_code=True, split=split),
        "format": format_arc,
        "val_split": "validation",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
    },
    "commonsenseqa": {
        "load": lambda split: load_dataset("tau/commonsense_qa", trust_remote_code=True, split=split),
        "format": format_commonsenseqa,
        "val_split": "validation",
        "test_split": "validation",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
    },
    "piqa": {
        "load": lambda split: load_dataset("ybisk/piqa", trust_remote_code=True, split=split),
        "format": format_piqa,
        "val_split": "validation",
        "test_split": "validation",
        "metric": "accuracy",
        "answer_extractor": "mcq_digit",
    },
    "winogrande": {
        "load": lambda split: load_dataset("allenai/winogrande", "winogrande_xl", trust_remote_code=True, split=split),
        "format": format_winogrande,
        "val_split": "validation",
        "test_split": "validation",
        "metric": "accuracy",
        "answer_extractor": "mcq_digit",
    },
    "boolq": {
        "load": lambda split: load_dataset("google/boolq", trust_remote_code=True, split=split),
        "format": format_boolq,
        "val_split": "validation",
        "test_split": "validation",
        "metric": "accuracy",
        "answer_extractor": "yesno",
    },
    "hellaswag": {
        "load": lambda split: load_dataset("Rowan/hellaswag", trust_remote_code=True, split=split),
        "format": format_hellaswag,
        "val_split": "validation",
        "test_split": "validation",
        "metric": "accuracy",
        "answer_extractor": "mcq_digit",
    },
    "math": {
        "load": _load_math,
        "format": format_math,
        "val_split": "test",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "math_boxed",
        "train_filter": _math_train_filter,
    },
    "openbookqa": {
        "load": lambda split: load_dataset("allenai/openbookqa", "main", split=split),
        "format": format_openbookqa,
        "val_split": "validation",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
    },
    "sciq": {
        "load": lambda split: load_dataset("allenai/sciq", split=split),
        "format": format_sciq,
        "val_split": "validation",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
    },
    "mbpp": {
        "load": _load_mbpp,
        "format": format_mbpp,
        "val_split": "test",
        "test_split": "test",
        "metric": "bleu_rouge",
        "answer_extractor": None,
    },
    "logiqa2": {
        "load": lambda split: load_dataset("baber/logiqa2", trust_remote_code=True, split=split),
        "format": format_logiqa2,
        "val_split": "validation",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
    },
    "drop": {
        "load": lambda split: load_dataset("ucinlp/drop", split=split),
        "format": format_drop,
        "val_split": "validation",
        "test_split": "validation",
        "metric": "accuracy",
        "answer_extractor": "drop_span",
        "train_filter": _long_passage_filter,
    },
    "mmlu_aux": {
        "load": lambda split: load_dataset("cais/mmlu", "auxiliary_train", split=split if split == "train" else "train"),
        "format": format_mmlu_aux,
        "val_split": "train",
        "test_split": "train",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
    },
    "triviaqa": {
        "load": lambda split: load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split),
        "format": format_triviaqa,
        "val_split": "validation",
        "test_split": "validation",
        "metric": "accuracy",
        "answer_extractor": "factoid_normalize",
        "max_train_samples": 80000,
        "train_filter": _long_passage_filter,
    },
    "anli": {
        "load": _load_anli,
        "format": format_anli,
        "val_split": "validation",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
        "max_train_samples": 30000,
    },
    "e2e": {
        "load": lambda split: load_dataset("e2e_nlg", trust_remote_code=True, split=split),
        "format": format_e2e,
        "val_split": "validation",
        "test_split": "test",
        "metric": "bleu_rouge",
        "answer_extractor": None,
    },
    "samsum": {
        "load": lambda split: load_dataset("knkarthick/samsum", trust_remote_code=True, split=split),
        "format": format_samsum,
        "val_split": "validation",
        "test_split": "test",
        "metric": "bleu_rouge",
        "answer_extractor": None,
    },
    "mmlu": {
        "load": lambda split: load_dataset("cais/mmlu", "all", split="test"),
        "format": format_mmlu,
        "val_split": "test",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
        "eval_only": True,
    },
    "mmlu_pro": {
        "load": lambda split: load_dataset("TIGER-Lab/MMLU-Pro", split="test"),
        "format": format_mmlu_pro,
        "val_split": "test",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
        "eval_only": True,
    },
    "bbh": {
        "load": _load_bbh,
        "format": format_bbh,
        "val_split": "test",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "exact_match_normalize",
        "eval_only": True,
    },
    "ifeval": {
        "load": lambda split: load_dataset("google/IFEval", split="train"),
        "format": format_ifeval,
        "val_split": "train",
        "test_split": "train",
        "metric": "bleu_rouge",
        "answer_extractor": None,
        "eval_only": True,
    },
    "agieval": {
        "load": lambda split: load_dataset("baber/agieval", trust_remote_code=True, split="test"),
        "format": format_agieval,
        "val_split": "test",
        "test_split": "test",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
        "eval_only": True,
    },
    "gpqa_diamond": {
        "load": lambda split: load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"),
        "format": format_gpqa_diamond,
        "val_split": "train",
        "test_split": "train",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
        "eval_only": True,
    },
    "truthfulqa": {
        "load": lambda split: load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation"),
        "format": format_truthfulqa,
        "val_split": "validation",
        "test_split": "validation",
        "metric": "accuracy",
        "answer_extractor": "mcq_letter",
        "eval_only": True,
    },
}


def load_and_prepare_dataset(dataset_name, tokenizer, max_seq_len, split="train"):
    info = REASONING_DATASET_LOADERS[dataset_name]
    actual_split = info["val_split"] if split == "validation" else split
    ds = info["load"](actual_split)
    fmt = info["format"]
    train_filter = info.get("train_filter") if split == "train" else None
    max_train_samples = info.get("max_train_samples") if split == "train" else None

    pairs = []
    dropped_filter = 0
    for ex in ds:
        try:
            ctx, ans = fmt(ex)
            if ans is None or ans == "" or ans == "0" or ctx == "":
                continue
            if train_filter is not None and not train_filter(ctx, ans):
                dropped_filter += 1
                continue
            pairs.append((ctx, ans))
        except (IndexError, KeyError, TypeError):
            continue

    if dropped_filter > 0:
        print(f"    [{dataset_name}] dropped {dropped_filter} train examples by length filter")

    if max_train_samples is not None and len(pairs) > max_train_samples:
        rng = random.Random(42 + hash(dataset_name) % 10000)
        pairs = rng.sample(pairs, max_train_samples)
        print(f"    [{dataset_name}] subsampled train to {max_train_samples} examples")

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
    info = REASONING_DATASET_LOADERS[dataset_name]

    # e2e and samsum have dedicated multi-reference loaders in data.py — use
    # those so BLEU/ROUGE-L uses all available references. Other bleu_rouge
    # datasets (mbpp, ifeval) fall through to the generic single-ref path
    # below; compute_metrics() normalizes flat refs to list-of-lists.
    if dataset_name in {"e2e", "samsum"}:
        from src.data import load_raw_dataset as load_raw_nlg
        return load_raw_nlg(dataset_name, split)

    actual_split = info["test_split"] if split == "test" else info["val_split"]
    ds = info["load"](actual_split)
    fmt = info["format"]

    inputs, references = [], []
    for ex in ds:
        try:
            ctx, ans = fmt(ex)
            if ans is None or ans == "" or ans == "0":
                continue
            inputs.append(ctx)
            references.append(ans)
        except (IndexError, KeyError, TypeError):
            continue

    return inputs, references
