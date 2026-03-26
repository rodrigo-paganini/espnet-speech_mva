"""
Prepare per-domain ASR data in ML-SUPERB format by streaming from Hugging Face.

Creates the directory structure expected by ML-SUPERB's data_prep.py:

    <output_root>/
      <dataset_name>/
        <lang3>/
          wav/
            <utt_id>.wav
          transcript_10min_train.txt
          transcript_1h_train.txt
          transcript_10min_dev.txt
          transcript_10min_test.txt

Transcript line format:  <utt_id> <duration_seconds> <text>
Utterance IDs:           <dataset_prefix>_<lang3>_<seq_number>
Sequence numbers:        [0, N)  where N = n_train + n_dev + n_test
                         ordered: train IDs first, then dev, then test.
"""

import argparse
import logging
import os
import string
import re
import json
import soundfile as sf
import numpy as np
from pathlib import Path
import tempfile
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DURATION_10MIN = 10 * 60   # seconds
DURATION_1H = 60 * 60
DURATION_DEV = DURATION_10MIN
DURATION_TEST = DURATION_10MIN

TARGET_SR = 16_000  # 16 kHz mono

# ISO 639-3 -> ISO 639-1 (for Hugging Face dataset configs)
LANG3_TO_LANG2 = {
    "eng": "en",
    "deu": "de",
    "fra": "fr",
    "spa": "es",
    "ita": "it",
    "nld": "nl",
    "pol": "pl",
}

# MLS uses full language names as config keys
LANG3_TO_MLS_CONFIG = {
    "eng": "english",
    "deu": "german",
    "fra": "french",
    "spa": "spanish",
    "ita": "italian",
    "nld": "dutch",
    "pol": "polish",
}

# YODAS manual-caption shard names (first shard per language, first digit 0 = manual)
LANG3_TO_YODAS_CONFIG = {
    "eng": "en000",
    "deu": "de000",
    "fra": "fr000",
    "spa": "es000",
    "ita": "it000",
    "nld": "nl000",
    "pol": "pl000",
}

DATASET_NAMES = [
    "mls",
    "voxpopuli",
    "commonvoice",
    "yodas"
]

# Which HF splits to use for our train / dev / test
HF_SPLIT_MAP = {
    "train": "train",
    "dev": "validation",
    "test": "test",
}

SPLIT_DURATION = {
"train": DURATION_1H,
"dev": DURATION_DEV,
"test": DURATION_TEST,
}

# Languages supported by each dataset
DATASET_LANGUAGES = ["eng", "deu", "fra", "spa", "ita", "nld", "pol"]

YODAS_LOADER_DIR = None

def get_yodas_local_loader() -> str:
    """
    Returns the path to a local directory containing patched YODAS dataset loading
    scripts. The directory is created once and reused for the process lifetime.
    """
    global YODAS_LOADER_DIR
    if YODAS_LOADER_DIR is not None:
        return YODAS_LOADER_DIR

    from huggingface_hub import hf_hub_download

    loader_dir = Path(tempfile.mkdtemp(prefix="yodas_loader_")) / "yodas"
    loader_dir.mkdir()

    # We download meta.py (lang -> shard-count mapping)
    meta_src = hf_hub_download(
        "espnet/yodas", "meta.py", repo_type="dataset"
    )
    shutil.copy2(meta_src, loader_dir / "meta.py")

    # We download yodas.py and patch base_data_path to use absolute HF URLs
    # so that dl_manager.download() resolves correctly from a local script.
    yodas_src = hf_hub_download(
        "espnet/yodas", "yodas.py", repo_type="dataset"
    )
    with open(yodas_src, encoding="utf-8") as f:
        script = f.read()

    HF_BASE = "https://huggingface.co/datasets/espnet/yodas/resolve/main"
    script = script.replace(
        'self.base_data_path = f"data/{lang}"',
        f'self.base_data_path = f"{HF_BASE}/data/{{lang}}"',
    )

    (loader_dir / "yodas.py").write_text(script, encoding="utf-8")

    YODAS_LOADER_DIR = str(loader_dir)
    log.info(f"YODAS local loader set up at {YODAS_LOADER_DIR}")
    return YODAS_LOADER_DIR

def process_text(text: str) -> str:
    """Normalizes text."""
    text = re.sub(r"\[[^\]]*\]", "", text)      # remove bracketed annotations
    text = re.sub(r"\([^)]*\)", "", text)        # remove parenthesized annotations
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper()

def load_hf_stream(dataset_name: str, lang3: str, hf_split: str):
    """
    Returns a Hugging Face streaming IterableDataset for the given
    (dataset, language, split) triplet and a function to extract (audio, text).

    Returns: (iterable_dataset, extract_fn)
        where extract_fn(example) -> (np.ndarray audio @ 16kHz, str text)
    """
    from datasets import load_dataset, Audio

    lang2 = LANG3_TO_LANG2[lang3]

    if dataset_name == "mls":
        split = "dev" if hf_split == "validation" else hf_split
        if lang3 == "eng":
            ds = load_dataset(
                "parler-tts/mls_eng",
                split=split,
                streaming=True,
            ).cast_column("audio", Audio(sampling_rate=TARGET_SR))
        else:
            config = LANG3_TO_MLS_CONFIG[lang3]
            ds = load_dataset(
                "facebook/multilingual_librispeech",
                config,
                split=split,
                streaming=True,
            ).cast_column("audio", Audio(sampling_rate=TARGET_SR))

        return ds, lambda ex: (ex["audio"]["array"], ex["transcript"])

    elif dataset_name == "voxpopuli":
        ds = load_dataset(
            "facebook/voxpopuli",
            lang2,
            split=hf_split,
            streaming=True,
        ).cast_column("audio", Audio(sampling_rate=TARGET_SR))

        return ds, lambda ex: (ex["audio"]["array"], (text := ex.get("normalized_text") or ex.get("raw_text", "")))

    elif dataset_name == "commonvoice":
        ds = load_dataset(
            "fsicoli/common_voice_22_0",
            lang2,
            split=hf_split,
            streaming=True,
            trust_remote_code=True
        ).cast_column("audio", Audio(sampling_rate=TARGET_SR))

        return ds, lambda ex: (ex["audio"]["array"], ex["sentence"])

    elif dataset_name == "yodas":
        config = LANG3_TO_YODAS_CONFIG[lang3]
        # YODAS only has a "train" split in HF; we handle dev/test by
        # skipping the first portion and using subsequent segments.
        loader_path = get_yodas_local_loader()
        ds = load_dataset(
            #"espnet/yodas",
            loader_path,
            config,
            split="train",
            streaming=True,
            trust_remote_code=True
        ).cast_column("audio", Audio(sampling_rate=TARGET_SR))

        return ds, lambda ex: (ex["audio"]["array"], ex["text"])

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def collect_utterances(
    stream_iter,
    extract_fn,
    target_duration: float,
    max_utt_dur: float,
    min_utt_dur: float = 0.5,
    min_text_len: int = 2,
) -> list:
    """
    Iterates through the HF stream and collects utterances until
    accumulated duration >= target_duration.

    Returns list of dicts: [{"audio": np.array, "text": str, "duration": float}, ...]
    """
    collected = []
    total_dur = 0.0

    for example in stream_iter:
        audio, text = extract_fn(example)

        if audio is None or len(audio) == 0:
            continue

        dur = len(audio) / TARGET_SR

        # We skip too long or too short audios
        if dur > max_utt_dur or dur < min_utt_dur:
            continue

        # We normalize and validate the text
        clean_text = process_text(text)
        if len(clean_text) < min_text_len:
            continue

        collected.append({
            "audio": audio,
            "text": clean_text,
            "duration": dur,
        })
        total_dur += dur

        if total_dur >= target_duration:
            break

    return collected

def write_transcript(path, utts, ids):
    with open(path, "w", encoding="utf-8") as f:
        for utt, utt_id in zip(utts, ids):
            f.write(f"{utt_id} {utt['duration']:.2f} {utt['text']}\n")


def process_dataset_language(
    dataset_name: str,
    lang3: str,
    output_root: Path,
    max_utt_dur: float,
    seed: int,
):
    """
    Streams data for one (dataset, language) pair, collects train/dev/test
    utterances, and writes them to disk in ML-SUPERB format.
    """
    prefix = dataset_name
    out_dir = output_root / dataset_name / lang3

    # Check if already done
    marker = out_dir / ".done"
    if marker.exists():
        log.info(f"  Skipping {dataset_name}/{lang3} (already done)")
        return

    log.info(f"  Processing {dataset_name}/{lang3} ...")

    # Splits collection
    # For YODAS there is only one HF split ("train"), so we collect
    # everything sequentially: first 1h for train, then 10min dev, then 10min test.
    # For other datasets we use separate HF splits.

    is_yodas = (dataset_name == "yodas")

    if is_yodas:
        # Single stream: we collect train+dev+test sequentially
        ds, extract_fn = load_hf_stream(dataset_name, lang3, "train")
        stream_iter = iter(ds)

        train_utts = collect_utterances(
            stream_iter, extract_fn, DURATION_1H, max_utt_dur
        )
        dev_utts = collect_utterances(
            stream_iter, extract_fn, DURATION_DEV, max_utt_dur
        )
        test_utts = collect_utterances(
            stream_iter, extract_fn, DURATION_TEST, max_utt_dur
        )
    else:
        # Separate HF splits
        splits_data = {}
        for our_split, hf_split in HF_SPLIT_MAP.items():
            target_dur = SPLIT_DURATION[our_split]
            try:
                ds, extract_fn = load_hf_stream(dataset_name, lang3, hf_split)
                utts = collect_utterances(
                    iter(ds), extract_fn, target_dur, max_utt_dur
                )
            except Exception as e:
                log.warning(
                    f"  Failed to load {dataset_name}/{lang3}/{hf_split}: {e}"
                )
                utts = []
            splits_data[our_split] = utts

        train_utts = splits_data["train"]
        dev_utts = splits_data["dev"]
        test_utts = splits_data["test"]

    train_dur = sum(u["duration"] for u in train_utts)
    dev_dur = sum(u["duration"] for u in dev_utts)
    test_dur = sum(u["duration"] for u in test_utts)

    log.info(
        f"    Collected: train={train_dur:.0f}s ({len(train_utts)} utts), "
        f"dev={dev_dur:.0f}s ({len(dev_utts)} utts), "
        f"test={test_dur:.0f}s ({len(test_utts)} utts)"
    )

    if len(train_utts) == 0 or len(dev_utts) == 0 or len(test_utts) == 0:
        log.warning(f"  Skipping {dataset_name}/{lang3}: insufficient data")
        return

    # Assigning utterance IDs
    # IDs are sequential in [0, N), ordered: train, dev, test
    N = len(train_utts) + len(dev_utts) + len(test_utts)

    train_start = 0
    dev_start = len(train_utts)
    test_start = dev_start + len(dev_utts)

    train_ids = [f"{prefix}_{lang3}_{i:06d}" for i in range(train_start, dev_start)]
    dev_ids = [f"{prefix}_{lang3}_{i:06d}" for i in range(dev_start, test_start)]
    test_ids = [f"{prefix}_{lang3}_{i:06d}" for i in range(test_start, N)]

    # We build the 10min train subset from the beginning of the 1h set
    train_10min_utts = []
    train_10min_ids = []
    accum = 0.0
    for utt, uid in zip(train_utts, train_ids):
        train_10min_utts.append(utt)
        train_10min_ids.append(uid)
        accum += utt["duration"]
        if accum >= DURATION_10MIN:
            break

    log.info(
        f"    10min train subset: {accum:.0f}s ({len(train_10min_utts)} utts)"
    )

    # We write all wav files
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    all_utts = list(zip(train_utts, train_ids)) + \
               list(zip(dev_utts, dev_ids)) + \
               list(zip(test_utts, test_ids))

    for utt, utt_id in all_utts:
        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            sf.write(str(wav_path), utt["audio"], TARGET_SR, subtype="PCM_16")

    # We write the transcripts' files
    write_transcript(out_dir / "transcript_1h_train.txt", train_utts, train_ids)
    write_transcript(out_dir / "transcript_10min_train.txt", train_10min_utts, train_10min_ids)
    write_transcript(out_dir / "transcript_10min_dev.txt", dev_utts, dev_ids)
    write_transcript(out_dir / "transcript_10min_test.txt", test_utts, test_ids)

    # We write a metadata file for bookkeeping
    meta = {
        "dataset": dataset_name,
        "language": lang3,
        "prefix": prefix,
        "train_1h_ids": train_ids,
        "train_10min_ids": train_10min_ids,
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "train_1h_duration": train_dur,
        "dev_duration": dev_dur,
        "test_duration": test_dur,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # We mark as done
    marker.touch()
    log.info(f"  Done: {dataset_name}/{lang3}")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare per-domain ASR data in ML-SUPERB format"
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
        help="Root directory for the output data",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["mls", "voxpopuli", "commonvoice", "yodas"],
        help="Datasets to prepare",
    )
    parser.add_argument(
        "--languages", type=str, nargs="+",
        default=["eng", "deu", "fra", "spa", "ita", "nld", "pol"],
        help="Languages (ISO 639-3 codes)",
    )
    parser.add_argument(
        "--max_utt_dur", type=float, default=20.0,
        help="Maximum utterance duration in seconds (default: 20.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (for any future shuffling)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_NAMES:
            log.error(f"Unknown dataset: {dataset_name}")
            continue

        supported_langs = DATASET_LANGUAGES
        log.info(f"=== Dataset: {dataset_name} ===")

        for lang3 in args.languages:
            if lang3 not in LANG3_TO_LANG2:
                log.warning(f"  Unknown language code: {lang3}, skipping")
                continue
            if lang3 not in supported_langs:
                log.info(f"  {lang3} not supported by {dataset_name}, skipping")
                continue

            try:
                process_dataset_language(
                    dataset_name, lang3, output_root,
                    args.max_utt_dur, args.seed,
                )
            except Exception as e:
                log.error(
                    f"  Error processing {dataset_name}/{lang3}: {e}",
                    exc_info=True,
                )

    log.info("All done.")


if __name__ == "__main__":
    main()