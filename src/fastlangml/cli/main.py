"""Command-line interface for fastlangml.

Usage:
    fastlangml "bonjour"              # Detect language, output JSON
    fastlangml "hello" --top-k 3      # Get top 3 languages
    fastlangml --version              # Show version
    fastlangml batch input.txt        # Batch process file
    fastlangml bench --dataset self   # Run benchmark
    fastlangml backends               # List backends
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def get_version() -> str:
    """Get package version."""
    try:
        from importlib.metadata import version

        return version("fastlangml")
    except Exception:
        return "0.1.0"


def detect_command(args: argparse.Namespace) -> int:
    """Handle language detection command."""
    from fastlangml import FastLangDetector

    detector = FastLangDetector()

    # Set allowed languages if specified
    if args.languages:
        detector.set_languages(args.languages.split(","))

    text = args.text

    # Read from stdin if text is "-"
    if text == "-":
        text = sys.stdin.read().strip()

    if not text:
        output = {"lang": "und", "reason": "empty_text"}
        print(json.dumps(output, indent=2 if args.pretty else None, ensure_ascii=False))
        return 0

    mode = "short" if args.short else "default"
    result = detector.detect(text, top_k=args.top_k, mode=mode)

    if args.include_meta:
        output = result.to_dict()
        output["text"] = text[:100] + "..." if len(text) > 100 else text
    else:
        output = {
            "lang": result.lang,
            "confidence": round(result.confidence, 4),
            "reliable": result.reliable,
        }
        if result.script:
            output["script"] = result.script
        if result.reason:
            output["reason"] = result.reason
        if args.top_k > 1 and result.candidates:
            output["candidates"] = [
                {"lang": c.lang, "confidence": round(c.confidence, 4)} for c in result.candidates
            ]
        output["text"] = text[:100] + "..." if len(text) > 100 else text

    if args.format == "table":
        print(f"Lang: {result.lang}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Reliable: {result.reliable}")
        if result.script:
            print(f"Script: {result.script}")
        if result.reason:
            print(f"Reason: {result.reason}")
    else:
        print(json.dumps(output, indent=2 if args.pretty else None, ensure_ascii=False))

    return 0


def batch_command(args: argparse.Namespace) -> int:
    """Handle batch detection command."""
    from fastlangml import FastLangDetector

    detector = FastLangDetector()

    if args.languages:
        detector.set_languages(args.languages.split(","))

    # Read lines from file or stdin
    if args.input == "-":
        lines = [line.strip() for line in sys.stdin if line.strip()]
    else:
        with open(args.input, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

    mode = "short" if args.short else "default"
    results = detector.detect_batch(lines, mode=mode)

    if args.format == "jsonl":
        for text, result in zip(lines, results, strict=True):
            output = {
                "text": text[:50],
                "lang": result.lang,
                "confidence": round(result.confidence, 4),
            }
            if result.reason:
                output["reason"] = result.reason
            print(json.dumps(output, ensure_ascii=False))
    else:
        output = [
            {
                "text": text[:50],
                "lang": result.lang,
                "confidence": round(result.confidence, 4),
            }
            for text, result in zip(lines, results, strict=True)
        ]
        print(json.dumps(output, indent=2 if args.pretty else None, ensure_ascii=False))

    # Write to output file if specified
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for text, result in zip(lines, results, strict=True):
                output = {
                    "text": text[:50],
                    "lang": result.lang,
                    "confidence": round(result.confidence, 4),
                }
                if result.reason:
                    output["reason"] = result.reason
                f.write(json.dumps(output, ensure_ascii=False) + "\n")

    return 0


def bench_command(args: argparse.Namespace) -> int:
    """Handle benchmark command."""
    from fastlangml import FastLangDetector

    detector = FastLangDetector()

    if args.dataset == "wili":
        return run_wili_benchmark(detector, args)
    else:
        return run_self_benchmark(detector, args)


def run_self_benchmark(detector: Any, args: argparse.Namespace) -> int:
    """Run self benchmark with built-in test cases."""
    # Self benchmark with built-in test cases
    test_cases = [
        ("Hello, how are you today?", "en"),
        ("Bonjour, comment allez-vous?", "fr"),
        ("Hola, como estas?", "es"),
        ("Guten Tag, wie geht es Ihnen?", "de"),
        ("Ciao, come stai?", "it"),
        ("Olá, como você está?", "pt"),
        ("Привет, как дела?", "ru"),
        ("こんにちは、お元気ですか？", "ja"),
        ("你好，你好吗？", "zh"),
        ("안녕하세요, 어떻게 지내세요?", "ko"),
        ("مرحبا، كيف حالك؟", "ar"),
        ("שלום, מה שלומך?", "he"),
        ("Γεια σου, πώς είσαι;", "el"),
        ("Merhaba, nasılsın?", "tr"),
        ("Xin chào, bạn có khỏe không?", "vi"),
    ]

    print("\n--- Self Benchmark ---")
    print(f"{'Text':<40} {'Expected':<8} {'Got':<8} {'Time (ms)':<12} Status")
    print("-" * 80)

    correct = 0
    total_time = 0.0

    for text, expected in test_cases:
        start = time.perf_counter()
        result = detector.detect(text)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        got = result.lang
        status = "✓" if got == expected else "✗"
        if got == expected:
            correct += 1

        text_display = text[:40] if len(text) <= 40 else text[:37] + "..."
        print(f"{text_display:<40} {expected:<8} {got:<8} {elapsed:>10.2f}ms {status}")

    print("-" * 80)
    print(f"Accuracy: {correct}/{len(test_cases)} ({correct / len(test_cases) * 100:.1f}%)")
    print(f"Avg time: {total_time / len(test_cases):.2f}ms")
    print(f"Total time: {total_time:.2f}ms")

    return 0 if correct == len(test_cases) else 1


# ISO 639-3 to ISO 639-1 mapping for common languages
# Also includes variant/dialect mappings
ISO3_TO_ISO1 = {
    # Major languages
    "eng": "en",
    "fra": "fr",
    "deu": "de",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "jpn": "ja",
    "zho": "zh",
    "kor": "ko",
    "ara": "ar",
    "heb": "he",
    "ell": "el",
    "tur": "tr",
    "vie": "vi",
    "nld": "nl",
    "pol": "pl",
    "ukr": "uk",
    "ces": "cs",
    "ron": "ro",
    "hun": "hu",
    "fin": "fi",
    "swe": "sv",
    "dan": "da",
    "nor": "no",
    "hin": "hi",
    "ben": "bn",
    "tam": "ta",
    "tel": "te",
    "mar": "mr",
    "urd": "ur",
    "fas": "fa",
    "tha": "th",
    "ind": "id",
    "msa": "ms",
    "cat": "ca",
    "eus": "eu",
    "glg": "gl",
    "bul": "bg",
    "hrv": "hr",
    "srp": "sr",
    "slk": "sk",
    "slv": "sl",
    "lit": "lt",
    "lav": "lv",
    "est": "et",
    "afr": "af",
    "swa": "sw",
    "amh": "am",
    "hau": "ha",
    "yor": "yo",
    "ibo": "ig",
    "mlt": "mt",
    "isl": "is",
    "cym": "cy",
    "gle": "ga",
    "gla": "gd",
    "bre": "br",
    "ltz": "lb",
    "bos": "bs",
    "mkd": "mk",
    "bel": "be",
    "kat": "ka",
    "hye": "hy",
    "aze": "az",
    "uzb": "uz",
    "kaz": "kk",
    "tgk": "tg",
    "mon": "mn",
    "mya": "my",
    "khm": "km",
    "lao": "lo",
    "sin": "si",
    "nep": "ne",
    "pus": "ps",
    "kur": "ku",
    "tuk": "tk",
    "tat": "tt",
    "uig": "ug",
    "yid": "yi",
    # Additional languages in WiLI-2018
    "wol": "wo",
    "rue": "uk",
    "pnb": "pa",
    "ori": "or",
    "mal": "ml",
    "kan": "kn",
    "guj": "gu",
    "pan": "pa",
    "asm": "as",
    "san": "sa",
    "bod": "bo",
    "div": "dv",
    "tir": "ti",
    "som": "so",
    "kin": "rw",
    "nya": "ny",
    "sna": "sn",
    "zul": "zu",
    "xho": "xh",
    "tsn": "tn",
    "ssw": "ss",
    "nso": "nso",
    "sot": "st",
    "nde": "nd",
    "lug": "lg",
    "lin": "ln",
    "kon": "kg",
    "orm": "om",
    "kik": "ki",
    "run": "rn",
    "fao": "fo",
    "hat": "ht",
    "cos": "co",
    "oci": "oc",
    "ast": "ast",
    "arg": "an",
    "lim": "li",
    "fry": "fy",
    "srd": "sc",
    "roh": "rm",
    "fur": "fur",
    "lad": "lad",
    "scn": "scn",
    "nap": "nap",
    "vec": "vec",
    "pms": "pms",
    "eml": "eml",
    "lmo": "lmo",
    "war": "war",
    "ceb": "ceb",
    "ilo": "ilo",
    "pag": "pag",
    "bcl": "bcl",
    "hil": "hil",
    "tgl": "tl",
    "jav": "jv",
    "sun": "su",
    "min": "min",
    "ace": "ace",
    "ban": "ban",
    "bjn": "bjn",
    "bug": "bug",
    "gor": "gor",
    "mak": "mak",
    "sas": "sas",
    "nds": "nds",
    "gsw": "gsw",
    "als": "als",
    "bar": "bar",
    "ksh": "ksh",
    "pfl": "pfl",
    "sgs": "sgs",
    "szl": "szl",
    "hsb": "hsb",
    "dsb": "dsb",
    "wln": "wa",
    "frr": "frr",
    "stq": "stq",
    "ang": "ang",
    "enm": "enm",
    "frm": "frm",
    "fro": "fro",
    "non": "non",
    "gmh": "gmh",
    "goh": "goh",
    "osx": "osx",
    "nrf": "nrf",
    "pcd": "pcd",
    "frp": "frp",
    "rup": "rup",
    "dlm": "dlm",
    "ext": "ext",
    "mwl": "mwl",
    "lld": "lld",
    "rgn": "rgn",
    "cbk": "cbk",
    "pap": "pap",
    "gcf": "gcf",
    "glv": "gv",
    "cor": "kw",
    "sco": "sco",
    "wym": "wym",
    "zlm": "ms",
    "zsm": "ms",
    "cmn": "zh",
    "yue": "zh",
    "wuu": "zh",
    "nan": "zh",
    "hak": "zh",
    "cdo": "zh",
    "gan": "zh",
    "hsn": "zh",
    "mnp": "zh",
    "cpx": "zh",
    "czh": "zh",
    "cjy": "zh",
    "csp": "zh",
    "och": "zh",
    "ltc": "zh",
    "lzh": "zh",
    "arz": "ar",
    "acm": "ar",
    "apc": "ar",
    "ary": "ar",
    "arb": "ar",
    "apd": "ar",
    "aao": "ar",
    "ajp": "ar",
    "aeb": "ar",
    "acw": "ar",
    "acx": "ar",
    "ayl": "ar",
    "ars": "ar",
    "prs": "fa",
    "pes": "fa",
    "ckb": "ku",
    "kmr": "ku",
    "sdh": "ku",
    "glk": "fa",
    "mzn": "mzn",
    "azb": "az",
    "crh": "crh",
    "tyv": "tyv",
    "sah": "sah",
    "xal": "xal",
    "kjh": "kjh",
    "cjs": "cjs",
    "krc": "krc",
    "kum": "kum",
    "nog": "nog",
    "gag": "gag",
    "chv": "cv",
    "bak": "ba",
    "udm": "udm",
    "koi": "koi",
    "kpv": "kv",
    "mhr": "mhr",
    "mrj": "mrj",
    "mdf": "mdf",
    "myv": "myv",
    "vep": "vep",
    "vro": "vro",
    "sme": "se",
    "smn": "smn",
    "sms": "sms",
    "smj": "smj",
    "sma": "sma",
    "liv": "liv",
    "krl": "krl",
    "olo": "olo",
    "izh": "izh",
    "vot": "vot",
    "sel": "sel",
    "yrk": "yrk",
    "nio": "nio",
    "kca": "kca",
    "mns": "mns",
    "ket": "ket",
    "ess": "ess",
    "esu": "esu",
    "ems": "ems",
    "evn": "evn",
    "eve": "eve",
    "neg": "neg",
    "oaa": "oaa",
    "ulc": "ulc",
    "ckt": "ckt",
    "itl": "itl",
    "ale": "ale",
    "chp": "chp",
    "dgr": "dgr",
    "gwi": "gwi",
    "hup": "hup",
    "nv": "nv",
    "nav": "nv",
    "mus": "mus",
    "cho": "cho",
    "chr": "chr",
    "moh": "moh",
    "oj": "oj",
    "oji": "oj",
    "cr": "cr",
    "cre": "cr",
    "iu": "iu",
    "iku": "iu",
    # Language variants (map to base language)
    "zh-yue": "zh",
    "zh-min-nan": "zh",
    "zh-classical": "zh",
    "be-tarask": "be",
    "pt-br": "pt",
    "en-gb": "en",
    "en-au": "en",
    "sr-latn": "sr",
    "sr-cyrl": "sr",
    "uz-latn": "uz",
    "uz-cyrl": "uz",
    "az-latn": "az",
    "az-cyrl": "az",
    "ku-arab": "ku",
    "ku-latn": "ku",
    "tg-cyrl": "tg",
    "tg-latn": "tg",
    "mn-cyrl": "mn",
    "mn-mong": "mn",
    # Fallback - some WiLI languages don't have ISO 639-1
    "xmf": "ka",
    "lzz": "ka",
    "bxr": "mn",
    "khk": "mn",
    "mvf": "mn",
}


def run_wili_benchmark(detector: Any, args: argparse.Namespace) -> int:
    """Run WiLI-2018 benchmark."""

    # Look for data in common locations
    data_paths = [
        Path("data"),
        Path("/Users/pankajrajan/fastlang/data"),
        Path.home() / ".fastlangml" / "data",
    ]

    data_dir = None
    for p in data_paths:
        if (p / "x_test.txt").exists():
            data_dir = p
            break

    if data_dir is None:
        print(
            json.dumps(
                {
                    "error": "WiLI-2018 dataset not found",
                    "message": "Download from: https://zenodo.org/records/841984/files/wili-2018.zip",
                    "expected_path": "data/x_test.txt",
                }
            )
        )
        return 1

    print(f"Loading WiLI-2018 dataset from {data_dir}...")

    # Load test data
    with open(data_dir / "x_test.txt", encoding="utf-8") as f:
        texts = [line.strip() for line in f]

    with open(data_dir / "y_test.txt", encoding="utf-8") as f:
        labels = [line.strip() for line in f]

    # Filter by languages if specified
    lang_filter = None
    if hasattr(args, "languages") and args.languages:
        lang_filter = set(args.languages.split(","))
        print(f"Filtering to languages: {lang_filter}")

    # Build filtered dataset
    filtered_texts = []
    filtered_labels = []
    for text, label in zip(texts, labels, strict=True):
        expected = ISO3_TO_ISO1.get(label, label[:2].lower())
        if lang_filter is None or expected in lang_filter:
            filtered_texts.append(text)
            filtered_labels.append(expected)

    n_samples = min(args.n_samples, len(filtered_texts))

    print(f"Running benchmark on {n_samples} samples...")
    print("-" * 80)

    correct = 0
    total_time = 0.0
    unknown_count = 0
    by_lang: dict[str, dict[str, int]] = {}  # lang -> {correct, total}

    for i in range(n_samples):
        text = filtered_texts[i]
        expected = filtered_labels[i]

        start = time.perf_counter()
        result = detector.detect(text, mode="default")
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        got = result.lang

        if expected not in by_lang:
            by_lang[expected] = {"correct": 0, "total": 0}
        by_lang[expected]["total"] += 1

        if got == "und":
            unknown_count += 1
        elif got == expected:
            correct += 1
            by_lang[expected]["correct"] += 1

        # Print progress every 100 samples
        if (i + 1) % 100 == 0:
            acc = correct / (i + 1 - unknown_count) * 100 if (i + 1 - unknown_count) > 0 else 0
            print(f"Processed {i + 1}/{n_samples} - Accuracy: {acc:.1f}% ({unknown_count} unknown)")

    print("-" * 80)
    evaluated = n_samples - unknown_count
    accuracy = correct / evaluated * 100 if evaluated > 0 else 0

    print("\n=== WiLI-2018 Benchmark Results ===")
    print(f"Samples: {n_samples}")
    print(f"Evaluated: {evaluated} (excluded {unknown_count} 'und')")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Avg time: {total_time / n_samples:.2f}ms per sample")
    print(f"Total time: {total_time / 1000:.2f}s")

    # Top 10 languages by accuracy
    print("\n--- Per-Language Accuracy (top 10) ---")
    lang_acc = [
        (lang, data["correct"] / data["total"] * 100, data["total"])
        for lang, data in by_lang.items()
        if data["total"] >= 5
    ]
    lang_acc.sort(key=lambda x: x[1], reverse=True)

    for lang, acc, total in lang_acc[:10]:
        print(f"  {lang}: {acc:.1f}% ({total} samples)")

    # Write results if output specified
    if args.out:
        results = {
            "dataset": "wili-2018",
            "samples": n_samples,
            "evaluated": evaluated,
            "correct": correct,
            "accuracy": accuracy,
            "unknown_count": unknown_count,
            "avg_time_ms": total_time / n_samples,
            "per_language": {
                lang: {"accuracy": data["correct"] / data["total"] * 100, "total": data["total"]}
                for lang, data in by_lang.items()
            },
        }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.out}")

    return 0


def backends_command(args: argparse.Namespace) -> int:
    """Handle backends listing command."""
    from fastlangml import get_available_backends
    from fastlangml.backends import BACKEND_RELIABILITY

    available = get_available_backends()
    all_backends = ["fasttext", "lingua", "pycld3", "langdetect", "langid"]

    output = {
        "available": available,
        "all_backends": all_backends,
        "reliability": {b: BACKEND_RELIABILITY.get(b, 0) for b in all_backends},
    }

    print(json.dumps(output, indent=2 if args.pretty else None))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="fastlangml",
        description="Fast, accurate language detection with multiple backends",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"fastlangml {get_version()}",
    )

    # Global options
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "jsonl", "table"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--pretty",
        "-p",
        action="store_true",
        help="Pretty-print JSON output",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Include metadata (timings, raw scores)",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Detect subcommand (also the default)
    detect_parser = subparsers.add_parser("detect", help="Detect language of text")
    detect_parser.add_argument("text", help="Text to detect (use '-' for stdin)")
    detect_parser.add_argument("--top-k", "-k", type=int, default=1)
    detect_parser.add_argument("--short", "-s", action="store_true")
    detect_parser.add_argument("--languages", "-l", type=str)
    detect_parser.add_argument("--format", "-f", choices=["json", "jsonl", "table"], default="json")
    detect_parser.add_argument("--pretty", "-p", action="store_true")
    detect_parser.add_argument("--include-meta", action="store_true")

    # Batch subcommand
    batch_parser = subparsers.add_parser("batch", help="Batch process file")
    batch_parser.add_argument("input", help="Input file or '-' for stdin")
    batch_parser.add_argument("--out", "-o", type=str, help="Output file (JSONL)")
    batch_parser.add_argument("--short", "-s", action="store_true")
    batch_parser.add_argument("--languages", "-l", type=str)
    batch_parser.add_argument("--format", "-f", choices=["json", "jsonl"], default="json")
    batch_parser.add_argument("--pretty", "-p", action="store_true")

    # Bench subcommand
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument("--dataset", "-d", choices=["wili", "self"], default="self")
    bench_parser.add_argument("--n-samples", "-n", type=int, default=1000, help="Number of samples")
    bench_parser.add_argument(
        "--languages", "-l", type=str, help="Filter languages (comma-separated)"
    )
    bench_parser.add_argument("--out", type=str, help="Output file for report")

    # Backends subcommand
    backends_parser = subparsers.add_parser("backends", help="List backends")
    backends_parser.add_argument("--pretty", "-p", action="store_true")

    # Handle case when called without subcommand (direct text)
    if argv is None:
        argv = sys.argv[1:]

    # Check if first arg looks like a command or text
    commands = ["detect", "batch", "bench", "backends", "-h", "--help", "-V", "--version"]
    if argv and argv[0] not in commands:
        # Assume it's text for detection - prepend "detect"
        argv = ["detect"] + argv

    args = parser.parse_args(argv)

    # Handle subcommands
    if args.command == "detect":
        return detect_command(args)
    elif args.command == "batch":
        return batch_command(args)
    elif args.command == "bench":
        return bench_command(args)
    elif args.command == "backends":
        return backends_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
