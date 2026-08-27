""" Training byte-level BPE tokenizers for RoBERTa or ModernBERT base models.

Uses the file-based ``tokenizer.train(files=[...])`` API: the Rust engine
streams the corpus from disk line-by-line, so Python-side RAM stays low.

Memory caveat: the trainer's internal word-frequency map holds one entry per
unique sequence. Because each DNA sequence is a single "word" under the
ByteLevel pre-tokenizer (no whitespace), a very large corpus can exceed
available RAM (e.g. ~8.5M sequences ~ 25-30 GB). If this OOMs, fall back to a
sample (train on a smaller file) or the iterator-based path with a cap.
"""
import argparse
import sys
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
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
    # The model config derives its vocab_size from len(tokenizer), so the
    # trainer target equals the configured vocab_size (special tokens included).
    vocab_size = SETTINGS["vocab_size"]

    print(f"Training tokenizer ({args.model}), vocab_size={vocab_size}")
    tokenizer = ByteLevelBPETokenizer()

    # Direct file-based training: the Rust engine streams the file from disk,
    # so Python RAM stays low (the trainer's word-frequency map is the only
    # significant memory consumer -- see module docstring caveat).
    tokenizer.train(
        files=[str(TRAIN_DATA)],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
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