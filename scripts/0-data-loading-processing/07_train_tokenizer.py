""" Training byte-level BPE tokenizers for RoBERTa or ModernBERT base models.

The corpus is streamed line-by-line from disk and fed to
``tokenizer.train_from_iterator`` so the entire file is never loaded into RAM
(this avoids OOM on large corpora, e.g. millions of promoter sequences).
"""
import argparse
import sys
from pathlib import Path

from tokenizers import ByteLevel, ByteLevelBPETokenizer, trainers
from transformers import PreTrainedTokenizerFast

# Make `module` importable when run as `python scripts/0-data-loading-processing/07_train_tokenizer.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from module.florabert import config


SETTINGS = config.settings["tokenizer"]  # previously config.settings["transformer"]["tokenizer"]

SPECIAL_TOKENS = {
    # RoBERTa-style special tokens (order determines token ids: <s>=0, </s>=1, ...)
    "roberta": ["<s>", "</s>", "<pad>", "<unk>", "<mask>"],
    # ModernBERT-style special tokens (order determines token ids: [CLS]=0, [SEP]=1, ...)
    "modernbert": ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"],
}


def iter_lines(path: Path, max_examples: int = None):
    """Lazily yield non-empty lines from `path`, optionally capping the count."""
    n = 0
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line
            n += 1
            if max_examples is not None and n >= max_examples:
                return


def main():
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer for RoBERTa or ModernBERT."
    )
    parser.add_argument(
        "--model",
        choices=["roberta", "modernbert"],
        default="roberta",
        help="Which special-token convention to use. 'modernbert' also saves the "
        "tokenizer as a fast-tokenizer directory (tokenizer.json) required by "
        "PreTrainedTokenizerFast (ModernBERT has no dedicated tokenizer class).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap the number of sequences used for training (None = use all). "
        "Useful to speed up tokenizer training on very large corpora.",
    )
    args = parser.parse_args()

    if args.model == "modernbert":
        TOKENIZER_DIR = config.modernbert_tokenizer_dir
    else:
        TOKENIZER_DIR = config.roberta_tokenizer_dir
    if not TOKENIZER_DIR.exists():
        TOKENIZER_DIR.mkdir()

    DATA_DIR = config.data_final / "transformer" / "seq"
    sample_data = DATA_DIR / "all_seqs_train_sample.txt"
    full_data = DATA_DIR / "all_seqs_train.txt"
    if sample_data.exists():
        TRAIN_DATA = sample_data
    elif full_data.exists():
        print(f"NOTE: {sample_data.name} not found, using {full_data.name}")
        TRAIN_DATA = full_data
    else:
        raise FileNotFoundError(
            f"No training sequences found under {DATA_DIR}. Expected either "
            f"{sample_data.name} or {full_data.name}."
        )
    special_tokens = SPECIAL_TOKENS[args.model]
    vocab_size = SETTINGS["vocab_size"] + len(special_tokens)

    print(f"Training tokenizer ({args.model}), vocab_size={vocab_size}, "
          f"max_examples={args.max_examples}")
    tokenizer = ByteLevelBPETokenizer()

    # Explicit byte-level BPE trainer (matches what ByteLevelBPETokenizer.train
    # uses), so we can drive it from a streaming iterator.
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        iter_lines(TRAIN_DATA, max_examples=args.max_examples), trainer=trainer
    )

    if args.model == "modernbert":
        print("Saving ModernBERT fast tokenizer")
        # ModernBERT (tokenizer_class "PreTrainedTokenizerFast") expects a full
        # fast-tokenizer directory (tokenizer.json) when loaded with
        # PreTrainedTokenizerFast.from_pretrained.
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="[UNK]",
            sep_token="[SEP]",
            cls_token="[CLS]",
            pad_token="[PAD]",
            mask_token="[MASK]",
        )
        fast_tokenizer.save_pretrained(str(TOKENIZER_DIR))
    else:
        print("Saving tokenizer")
        tokenizer.save_model(str(TOKENIZER_DIR))

    print("Done")


if __name__ == "__main__":
    main()