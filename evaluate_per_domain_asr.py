"""
Evaluate ASR results with per-domain breakdown.

After running the ML-SUPERB ASR pipeline (training + inference), this script
reads the reference and hypothesis text files and reports CER overall and
broken down by domain (dataset) and by language.

The domain is identified from the utterance ID prefix:
    mls_eng_000042      -> domain = mls
    voxpopuli_fra_000003 -> domain = voxpopuli
    cv_deu_000010       -> domain = commonvoice
    yodas_spa_000007    -> domain = yodas
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import editdistance

PREFIX_TO_DOMAIN = {
    "mls": "read_speech",
    "voxpopuli": "parliamentary",
    "commonvoice": "noisy_crowdsourced",
    "yodas": "conversational",
}

DOMAIN_TO_DATASET = {
    "read_speech": "mls",
    "parliamentary": "voxpopuli",
    "noisy_crowdsourced": "commonvoice",
    "conversational": "yodas",
}

def parse_utt_id(utt_id: str):
    """
    Parses an utterance ID into (domain, language).

    Examples:
        mls_eng_000042       -> ("read_speech", "eng")
        voxpopuli_fra_000003 -> ("parliamentary", "fra")
        commonvoice_deu_000010        -> ("noisy_crowdsourced", "deu")
        yodas_spa_000007     -> ("conversational", "spa")
    """
    parts = utt_id.split("_")

    return PREFIX_TO_DOMAIN[parts[0]], parts[1]

def read_text_file(path: str) -> dict:
    """
    Reads a Kaldi-style text file.
    Returns dict: utt_id -> text
    """
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt_id = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            result[utt_id] = text
    return result

def compute_cer(ref: str, hyp: str) -> tuple:
    """
    Computes character error rate between reference and hypothesis.
    Returns (edit_distance, ref_length).
    """
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))

    if len(ref_chars) == 0:
        return len(hyp_chars), 0

    dist = editdistance.eval(ref_chars, hyp_chars)
    return dist, len(ref_chars)

def evaluate(ref_dict: dict, hyp_dict: dict) -> dict:
    """
    Computes overall and per-domain/per-language CER.

    Returns a structured dict with all results.
    """
    # Accumulators: key -> (total_edits, total_ref_len, count)
    overall = {"edits": 0, "ref_len": 0, "count": 0}
    by_domain = defaultdict(lambda: {"edits": 0, "ref_len": 0, "count": 0})
    by_language = defaultdict(lambda: {"edits": 0, "ref_len": 0, "count": 0})
    by_domain_lang = defaultdict(lambda: {"edits": 0, "ref_len": 0, "count": 0})

    missing_hyp = 0

    for utt_id, ref_text in ref_dict.items():
        if utt_id not in hyp_dict:
            missing_hyp += 1
            continue

        hyp_text = hyp_dict[utt_id]
        domain, lang = parse_utt_id(utt_id)

        edits, ref_len = compute_cer(ref_text, hyp_text)

        overall["edits"] += edits
        overall["ref_len"] += ref_len
        overall["count"] += 1

        by_domain[domain]["edits"] += edits
        by_domain[domain]["ref_len"] += ref_len
        by_domain[domain]["count"] += 1

        by_language[lang]["edits"] += edits
        by_language[lang]["ref_len"] += ref_len
        by_language[lang]["count"] += 1

        key = f"{domain}/{lang}"
        by_domain_lang[key]["edits"] += edits
        by_domain_lang[key]["ref_len"] += ref_len
        by_domain_lang[key]["count"] += 1

    def cer_pct(acc):
        if acc["ref_len"] == 0:
            return float("nan")
        return 100.0 * acc["edits"] / acc["ref_len"]

    results = {
        "overall": {
            "cer": cer_pct(overall),
            "utterances": overall["count"],
            "ref_chars": overall["ref_len"],
        },
        "per_domain": {},
        "per_language": {},
        "per_domain_language": {},
        "missing_hypotheses": missing_hyp,
    }

    for domain in sorted(by_domain.keys()):
        acc = by_domain[domain]
        results["per_domain"][domain] = {
            "cer": cer_pct(acc),
            "utterances": acc["count"],
            "ref_chars": acc["ref_len"],
        }

    for lang in sorted(by_language.keys()):
        acc = by_language[lang]
        results["per_language"][lang] = {
            "cer": cer_pct(acc),
            "utterances": acc["count"],
            "ref_chars": acc["ref_len"],
        }

    for key in sorted(by_domain_lang.keys()):
        acc = by_domain_lang[key]
        results["per_domain_language"][key] = {
            "cer": cer_pct(acc),
            "utterances": acc["count"],
            "ref_chars": acc["ref_len"],
        }

    return results


def print_report(results: dict):
    """Prints a human-readable report of the evaluation results."""

    print("=" * 70)
    print("  PER-DOMAIN ASR EVALUATION REPORT")
    print("=" * 70)

    ov = results["overall"]
    print(f"\n  Overall CER: {ov['cer']:.2f}%  "
          f"({ov['utterances']} utterances, {ov['ref_chars']} ref chars)")

    if results["missing_hypotheses"] > 0:
        print(f"  WARNING: {results['missing_hypotheses']} reference utterances "
              f"had no hypothesis")

    print("\n" + "-" * 70)
    print("  PER DOMAIN")
    print("-" * 70)
    print(f"  {'Domain':<25} {'CER':>8} {'Utts':>8} {'Ref chars':>10}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 10}")

    domain_cers = {}
    for domain, info in sorted(results["per_domain"].items()):
        print(f"  {domain:<25} {info['cer']:>7.2f}% {info['utterances']:>8} "
              f"{info['ref_chars']:>10}")
        domain_cers[domain] = info["cer"]

    if len(domain_cers) >= 2:
        best = min(domain_cers.values())
        worst = max(domain_cers.values())
        print(f"\n  Degradation gap (worst - best): {worst - best:.2f}%")
        print(f"  Best domain:  {min(domain_cers, key=domain_cers.get)} "
              f"({best:.2f}%)")
        print(f"  Worst domain: {max(domain_cers, key=domain_cers.get)} "
              f"({worst:.2f}%)")

    print("\n" + "-" * 70)
    print("  PER LANGUAGE")
    print("-" * 70)
    print(f"  {'Language':<25} {'CER':>8} {'Utts':>8} {'Ref chars':>10}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 10}")

    for lang, info in sorted(results["per_language"].items()):
        print(f"  {lang:<25} {info['cer']:>7.2f}% {info['utterances']:>8} "
              f"{info['ref_chars']:>10}")

    print("\n" + "-" * 70)
    print("  PER DOMAIN AND LANGUAGE")
    print("-" * 70)
    print(f"  {'Domain/Language':<35} {'CER':>8} {'Utts':>8}")
    print(f"  {'─' * 35} {'─' * 8} {'─' * 8}")

    for key, info in sorted(results["per_domain_language"].items()):
        print(f"  {key:<35} {info['cer']:>7.2f}% {info['utterances']:>8}")

    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR results with per-domain and per-language breakdown"
    )
    parser.add_argument(
        "--ref_text", type=str, default=None,
        help="Path to reference text file (Kaldi format)",
    )
    parser.add_argument(
        "--hyp_text", type=str, default=None,
        help="Path to hypothesis text file (Kaldi format)",
    )
    parser.add_argument(
        "--scoring_dir", type=str, default=None,
        help="Alternative: path to ESPnet scoring directory containing "
             "ref.trn and hyp.trn (not used if --ref_text/--hyp_text given)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results (optional)",
    )
    args = parser.parse_args()

    # We load reference and hypothesis texts
    if args.ref_text and args.hyp_text:
        ref_dict = read_text_file(args.ref_text)
        hyp_dict = read_text_file(args.hyp_text)
    elif args.scoring_dir:
        # ESPnet stores scoring results in ref.trn / hyp.trn format:
        #   text (utt_id)
        scoring_dir = Path(args.scoring_dir)

        ref_dict = {}
        hyp_dict = {}

        ref_path = scoring_dir / "ref.trn"
        hyp_path = scoring_dir / "hyp.trn"

        if not ref_path.exists() or not hyp_path.exists():
            print("ref.trn / hyp.trn not found in scoring_dir.")
            print("Please provide --ref_text and --hyp_text instead.")
            sys.exit(1)

        # Parse .trn format: "text (utt_id)"
        for line in open(ref_path, "r", encoding="utf-8"):
            line = line.strip()
            m = re.match(r"^(.*)\((\S+)\)\s*$", line)
            if m:
                text, utt_id = m.group(1).strip(), m.group(2)
                ref_dict[utt_id] = text

        for line in open(hyp_path, "r", encoding="utf-8"):
            line = line.strip()
            m = re.match(r"^(.*)\((\S+)\)\s*$", line)
            if m:
                text, utt_id = m.group(1).strip(), m.group(2)
                hyp_dict[utt_id] = text
    else:
        print("Provide either --ref_text and --hyp_text, or --scoring_dir")
        sys.exit(1)

    print(f"Loaded {len(ref_dict)} reference utterances, "
          f"{len(hyp_dict)} hypotheses")

    results = evaluate(ref_dict, hyp_dict)

    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()