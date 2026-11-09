# FastLangML

[![PyPI](https://img.shields.io/pypi/v/fastlangml)](https://pypi.org/project/fastlangml/)
[![Python](https://img.shields.io/pypi/pyversions/fastlangml)](https://pypi.org/project/fastlangml/)
[![Tests](https://github.com/pnrajan/fastlangml/actions/workflows/tests.yml/badge.svg)](https://github.com/pnrajan/fastlangml/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Language detection for short, messy text: chat messages, SMS, support tickets.

FastLangML runs several detectors ([fastText](https://fasttext.cc/), [Lingua](https://github.com/pemistahl/lingua-py), [langdetect](https://github.com/Mimino666/langdetect), [langid](https://github.com/saffsd/langid.py), [CLD3](https://github.com/bsolomon1124/pycld3), [fastlangid](https://pypi.org/project/fastlangid/)) and combines their votes. On top of that it adds the things chat text needs:

- **Hint dictionaries** for slang and abbreviations (`thx`, `mdr`, `jaja`)
- **Conversation context** so earlier messages can inform later ones
- **Code-switching detection** for mixed messages (Spanglish, Hinglish, ...)
- **Confusable-pair handling** (Spanish/Portuguese, Norwegian/Danish/Swedish, ...)
- **An honest "unknown"**: returns `und` with a reason instead of guessing on empty or too-short input
- **Plug-in points** for your own backends and voting strategies

```python
from fastlangml import detect

detect("Bonjour") == "fr"   # True
```

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Command line](#command-line)
- [Conversations](#conversations)
- [Configuring the detector](#configuring-the-detector)
- [Advanced features](#advanced-features)
- [API reference](#api-reference)
- [Performance](#performance)
- [Development](#development)

---

## Install

FastLangML has no detection backend by default. Install at least one:

```bash
pip install "fastlangml[all]"               # every backend and optional feature
pip install "fastlangml[fasttext,lingua]"   # a good, smaller combination
```

Include **fastText** if you detect single words or very short messages. On a set of common greetings ("Hello", "Hola", "Ciao", "Merci", ...), `fasttext` + `lingua` got 8 of 8 right. `lingua`, `langid` and `langdetect` without fastText got 3 of 8. All backends do well on full sentences.

| Extra | Backend | Languages | Notes |
|-------|---------|-----------|-------|
| `fasttext` | fastText (`lid.176`) | 176 | Fast, broad coverage. Pins `numpy<2`. |
| `lingua` | Lingua | 75 | Most accurate on short text; slower. |
| `fastlangid` | fastlangid | 177 | Better on Chinese, Japanese, Korean, Cantonese. |
| `langdetect` | langdetect | 55 | |
| `langid` | langid.py | 97 | |
| `pycld3` | Google CLD3 | 107 | Needs `protobuf` system libraries to build. |

Optional features:

| Extra | Adds |
|-------|------|
| `diskcache` | `DiskContextStore` — save conversation context to disk |
| `redis` | `RedisContextStore` — save conversation context to Redis |
| `spacy` | spaCy named-entity filtering of proper nouns |
| `fuzzy` | Fuzzy matching for hint dictionaries |

Check what is installed:

```python
from fastlangml import FastLangDetector

FastLangDetector().available_backends   # e.g. ['fasttext', 'lingua', 'langdetect']
```

---

## Quick start

```python
from fastlangml import detect

result = detect("Hello, how are you?")

result.lang         # "en"
result.confidence   # 1.0
result.reliable     # True
result.backend      # "ensemble"
```

A `DetectionResult` compares and prints like its language code, so you can use it as a string:

```python
detect("Hello") == "en"        # True
str(detect("Hola"))            # "es"
f"{detect('Ciao')}"            # "it"
```

When there is nothing to detect, you get `und` and a reason:

```python
detect("").lang, detect("").reason             # ("und", "empty_text")
detect("12345").lang, detect("12345").reason   # ("und", "too_little_text")
```

More options:

```python
detect("Bonjour le monde", top_k=3).candidates   # ranked alternatives
detect("Hello", allowed_langs=["en", "fr"])      # only consider these languages
detect("Some longer paragraph ...", mode="long") # stricter threshold

from fastlangml import detect_batch
detect_batch(["Hello there", "Bonjour à tous"])  # list of results
```

`confidence` is the winning language's share of the combined backend vote: 1.0 when every backend agrees, about 0.5 when two languages split the vote. `mode` sets the minimum text length and how much of the vote the winner needs. Below that, you get `und` with reason `low_confidence` (or `disagreement` when the top two are close).

| Mode | Use for | Confidence threshold |
|------|---------|----------------------|
| `"short"` | Chat, SMS, single words (default for `detect()`) | 0.3 |
| `"default"` | Sentences | 0.5 |
| `"long"` | Paragraphs, documents | 0.6 |

---

## Command line

Installing the package adds a `fastlangml` command.

```bash
fastlangml detect "Bonjour le monde"
# {"lang":"fr","confidence":1.0,"reliable":true,"script":"Latin","text":"Bonjour le monde"}

echo "Hola amigo" | fastlangml detect -          # read from stdin
fastlangml detect --top-k 3 --short "ok"
fastlangml -f table "Guten Morgen zusammen"        # json | jsonl | table
fastlangml "Bonjour"                              # `detect` is the default command
fastlangml batch messages.txt                    # one text per line
fastlangml backends                              # list installed backends
fastlangml bench --dataset wili --n-samples 1000 # accuracy benchmark
```

Output options (`-f/--format`, `-p/--pretty`, `--include-meta`) work before or after the command. Run `fastlangml <command> --help` for all flags.

---

## Conversations

### Context

Pass a `ConversationContext` to remember the languages of recent messages. By default, `detect()` adds each result to the context for you.

```python
from fastlangml import ConversationContext, detect

ctx = ConversationContext(max_turns=10)

for message in ["Bonjour", "J'ai un problème avec ma commande", "ok", "merci"]:
    detect(message, context=ctx)

ctx.dominant_language       # "fr"
ctx.language_distribution   # mostly "fr", e.g. {"fr": 0.84, "en": 0.16}
```

How it works: the conversation votes alongside the backends. Each language's vote comes from how often it appeared in recent turns (newer turns count more, set by `decay_factor`). How much that vote counts depends on how much the text says on its own:

- **Full sentences** (20+ letters): context gets `0.3 × context_weight` of the vote (0.06 with defaults). It only breaks near-ties.
- **Short replies** ("Bien", "Gut", "merci"): context counts for more, up to 45% of the vote for words of 4 letters or fewer. It never reaches half, so if every backend agrees on another language, the backends win.
- **Too short to read** ("ok", "si"): you get the conversation's main language instead of `und`.

When the conversation picks the answer and the score is below the mode's threshold, the result has `reliable=False` and `meta["from_context"] = True`, and it is not added back to the context. To make context count for more on longer text, raise `DetectionConfig.context_weight`.

A common use is routing a whole conversation by `ctx.dominant_language` rather than trusting any single short message.

> **Note:** `detect()` updates the context by default (`auto_update=True`). `FastLangDetector.detect()` does **not** (`auto_update=False`). Pass `auto_update=True`, or call `ctx.add_turn(text, result.lang, result.confidence)` yourself.
>
> Undetermined results (`und`) are never added to the context, so empty or noisy messages don't push real history out.

### Saving context between requests

For web apps, where each message arrives in a separate request, save the context per session.

```python
from fastlangml import detect
from fastlangml.context import DiskContextStore    # pip install "fastlangml[diskcache]"

store = DiskContextStore("./contexts", ttl_seconds=3600)

with store.session("user-123") as ctx:              # loads, then saves on exit
    detect("Bonjour tout le monde", context=ctx)
```

For production, use Redis. The interface is the same:

```python
from fastlangml.context import RedisContextStore   # pip install "fastlangml[redis]"

store = RedisContextStore("redis://localhost:6379", ttl_seconds=1800)
```

Both stores take `max_turns` (default 5). To use your own storage instead, serialize the context:

```python
data = ctx.to_dict()                                   # JSON-safe dict
ctx = ConversationContext.from_dict(data)
ctx = ConversationContext.from_history([("fr", 0.95), ("fr", 0.9)])
```

---

## Configuring the detector

`detect()` uses a shared default detector. To configure one, create a `FastLangDetector`:

```python
from fastlangml import DetectionConfig, FastLangDetector

detector = FastLangDetector(
    config=DetectionConfig(
        backends=["fasttext", "lingua", "langdetect"],   # empty = use all installed
        backend_weights={"fasttext": 0.5, "lingua": 0.35, "langdetect": 0.15},
        voting_strategy="weighted",
    )
)

detector.detect("Ciao, come stai?")   # "it"
```

### Voting strategies

| Strategy | How it combines backends |
|----------|--------------------------|
| `"weighted"` (default) | Sum of confidence × backend weight |
| `"hard"` | One vote per backend; ties broken by confidence |
| `"soft"` | Average of backend probabilities |
| `"consensus"` | Only answers when enough backends agree |

```python
from fastlangml import ConsensusVoting

FastLangDetector(config=DetectionConfig(voting_strategy="hard"))
FastLangDetector(config=DetectionConfig(custom_voting=ConsensusVoting(min_agreement=2)))
```

### Hints

Hints map words to languages. Backends are trained on full sentences and often miss chat slang, so hints fill the gap. Built-in chat hints are **not** enabled by default; load them explicitly:

```python
from fastlangml import FastLangDetector, HintDictionary

detector = FastLangDetector(hints=HintDictionary.default_short_words())

detector.detect("thx")    # "en"
detector.detect("mdr")    # "fr"  (mort de rire, "LOL")
detector.detect("jaja")   # "es"
```

Add your own:

```python
detector.add_hint("stp", "fr")    # s'il te plaît
detector.detect("stp")            # "fr"
detector.remove_hint("stp")
```

In `"short"` mode, text of 5 characters or fewer is mostly decided by hints. For longer text, hints add a boost scaled by `hint_weight`.

### Restricting languages

If you know the possible answers, for example a bilingual support queue:

```python
detector.set_languages(["en", "es"])
```

### All `DetectionConfig` options

| Option | Default | Meaning |
|--------|---------|---------|
| `mode` | `"default"` | Default mode for `FastLangDetector.detect()` |
| `backends` | `[]` | Backends to use; empty means all installed |
| `backend_weights` | `{}` | Per-backend weight for weighted voting |
| `voting_strategy` | `"weighted"` | `"weighted"`, `"hard"`, `"soft"`, `"consensus"` |
| `custom_voting` | `None` | A `VotingStrategy` instance; overrides `voting_strategy` |
| `thresholds` | `{"short": 0.3, "default": 0.5, "long": 0.6}` | Share of the vote the winner needs; below it the result is `und` |
| `min_chars` | `{"short": 2, "default": 3, "long": 3}` | Shorter text returns `und` |
| `min_letters` | `{"short": 1, "default": 2, "long": 2}` | Fewer letters returns `und` |
| `min_letter_ratio` | `0.3` | Minimum share of letters in the text |
| `unknown_policy` | `{"allow": True, "label": "und"}` | Whether and how to return "unknown" |
| `normalize` | `True` | Unicode normalization |
| `strip_noise` | `True` | Remove URLs, emails, and similar before detecting |
| `filter_proper_nouns` | `True` | Remove likely names before detecting |
| `proper_noun_strategy` | `"remove"` | `"remove"`, `"mask"`, or `"none"` |
| `use_script_filter` | `True` | Use the writing system (Hangul, Thai, ...) to answer early |
| `cache_size` | `1000` | LRU cache size for repeated texts |
| `hint_weight` | `0.3` | Strength of hint boosts |
| `context_weight` | `0.2` | How much conversation history counts on full sentences (short text gets more) |

---

## Advanced features

### Code-switching

Detect messages that mix languages:

```python
from fastlangml import CodeSwitchDetector

cs = CodeSwitchDetector()
result = cs.detect("That's muy importante for the proyecto")

result.is_mixed                # True
result.primary_language        # "en"
result.secondary_languages     # ["es"]
result.language_distribution   # {"en": 0.6, "es": 0.4}

for span in result.spans:
    print(span.text, span.language, span.confidence)

cs.is_code_switched("Hello world")          # False
cs.is_code_switched("Hola, how are you?")   # True
```

### Confusable languages

Some languages are hard to tell apart. `ConfusionResolver` uses distinguishing words to adjust scores for these pairs:

Spanish/Portuguese · Norwegian/Danish/Swedish · Czech/Slovak · Croatian/Serbian/Bosnian · Indonesian/Malay · Russian/Ukrainian/Belarusian · Hindi/Urdu

```python
from fastlangml import ConfusionResolver, LanguageSimilarity

resolver = ConfusionResolver()
resolver.resolve("Eu tenho um problema", {"es": 0.45, "pt": 0.42})
# {"es": 0.45, "pt": 0.62}   Portuguese boosted ("tenho")

LanguageSimilarity().are_related("es", "pt")   # True
```

### Custom backends

```python
from fastlangml import Backend, DetectionConfig, FastLangDetector, backend
from fastlangml.backends.base import DetectionResult


@backend("my_detector", reliability=4)   # reliability: 1 (low) to 5 (high)
class MyBackend(Backend):
    @property
    def name(self) -> str:
        return "my_detector"

    @property
    def is_available(self) -> bool:
        return True   # e.g. check that your dependency imports

    def detect(self, text: str) -> DetectionResult:
        return DetectionResult(self.name, "en", 0.95)

    def supported_languages(self) -> set[str]:
        return {"en", "fr"}


detector = FastLangDetector(config=DetectionConfig(backends=["my_detector", "fasttext"]))
```

You can also register without the decorator: `register_backend(name, cls, reliability=3)`, `unregister_backend(name)`, `list_registered_backends()`.

Note that `fastlangml.backends.base.DetectionResult` is a backend's raw vote (`backend_name`, `language`, `confidence`). It is different from `fastlangml.DetectionResult`, the final result returned by `detect()`.

### Custom voting

```python
from fastlangml import VotingStrategy


class HighestConfidence(VotingStrategy):
    """Score each language by its single most confident backend."""

    def vote(self, results, weights=None) -> dict[str, float]:
        scores: dict[str, float] = {}
        for r in results:
            scores[r.language] = max(scores.get(r.language, 0.0), r.confidence)
        return scores


detector = FastLangDetector(config=DetectionConfig(custom_voting=HighestConfidence()))
```

---

## API reference

### `detect()`

```python
detect(
    text: str,
    *,
    context: ConversationContext | None = None,
    mode: Literal["short", "default", "long"] = "short",
    top_k: int = 1,
    allowed_langs: list[str] | None = None,
    auto_update: bool = True,
) -> DetectionResult
```

### `FastLangDetector`

```python
detector = FastLangDetector(config: DetectionConfig | None = None, hints: HintDictionary | None = None)

detector.detect(text, *, top_k=1, mode=None, allowed_langs=None,
                filter_proper_nouns=None, context=None, hints=None, auto_update=False)
detector.detect_batch(texts, mode=...)
detector.add_hint(word, lang)
detector.remove_hint(word)
detector.set_languages(["en", "fr"])
detector.available_backends    # property: list of installed backends
detector.cache_stats           # property: {"size": ..., "maxsize": ...}
```

### `DetectionResult`

| Field | Type | Meaning |
|-------|------|---------|
| `lang` | `str` | ISO 639-1 code, or `"und"` |
| `confidence` | `float` | Winner's share of the backend vote, 0.0 – 1.0 |
| `reliable` | `bool` | `confidence` ≥ the mode's threshold |
| `reason` | `str \| None` | Why the result is `und` |
| `script` | `str \| None` | Writing system, e.g. `"Latin"` |
| `backend` | `str` | Backend name, or `"ensemble"` |
| `candidates` | `list[Candidate]` | Ranked alternatives when `top_k > 1` |
| `meta` | `dict` | Timing and per-backend scores |

### `ConversationContext`

```python
ctx = ConversationContext(max_turns=2, decay_factor=0.9)

ctx.add_turn(text, detected_language, confidence)
ctx.dominant_language        # most likely language, or None
ctx.language_distribution    # {lang: weight}, sums to 1.0
ctx.last_turn
ctx.turns
ctx.get_language_streak()    # (lang, count)
ctx.clear()
ctx.to_dict() / ConversationContext.from_dict(d) / ConversationContext.from_history(pairs)
```

---

## Performance

- **Cache:** repeated texts are served from an LRU cache (`cache_size`). Calls that pass a `context` or per-call `hints` skip the cache, since those can change the result. `add_hint()` and `remove_hint()` clear the cache.
- **Script short-circuit:** text in a writing system used by only one language (Korean, Thai, Hebrew, Armenian, Georgian, Tamil, Telugu, Kannada, Malayalam, Gujarati, Bengali, Punjabi, Oriya, Sinhala, Khmer, Lao, Myanmar, Tibetan) is answered without running any backend.
- **Lazy loading:** backends load their models on first use. Expect the first call to take a few seconds; later calls take milliseconds.
- **Batching:** `detect_batch()` runs texts in parallel and is faster than calling `detect()` in a loop. Pass `max_workers=` to cap the number of threads.

Run the benchmark on your own data and hardware:

```bash
fastlangml bench --dataset wili --n-samples 1000 --out report.json
pytest tests/test_benchmarks.py -v
```

---

## Development

```bash
git clone https://github.com/pnrajan/fastlangml.git
cd fastlangml

poetry install --all-extras     # or: pip install -e ".[dev,all]"

make test        # pytest
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy
make check       # all of the above
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture, testing, and pull request guidelines.

## License

MIT — see [LICENSE](LICENSE).
