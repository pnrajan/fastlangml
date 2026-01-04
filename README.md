# FastLangML

[![PyPI](https://img.shields.io/pypi/v/fastlangml)](https://pypi.org/project/fastlangml/)
[![Python](https://img.shields.io/pypi/pyversions/fastlangml)](https://pypi.org/project/fastlangml/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-accuracy language detection for chat, SMS, and conversational text.** FastLangML combines multiple detection backends ([FastText](https://fasttext.cc/), [Lingua](https://github.com/pemistahl/lingua-py), [langdetect](https://github.com/Mimino666/langdetect), [pyCLD3](https://github.com/bsolomon1124/pycld3)) into a powerful ensemble that outperforms any single detector.

**Key Features:**
- **170+ languages** supported via multi-backend ensemble
- **Context-aware detection** - tracks conversation history to resolve ambiguous short messages like "ok", "si", "bien"
- **Code-switching detection** - identifies mixed-language messages (Spanglish, Franglais, Hinglish)
- **Slang & abbreviations** - built-in hints for chat lingo ("thx", "mdr", "jaja")
- **Confusion resolution** - handles similar language pairs (Spanish/Portuguese, Norwegian/Danish/Swedish)
- **Extensible** - add custom backends, voting strategies, and hint dictionaries

## Table of Contents

- [The Problem](#the-problem)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Context-Aware Detection](#context-aware-detection)
  - [Multi-Backend Ensemble](#multi-backend-ensemble)
  - [Voting Strategies](#voting-strategies)
  - [Hint Dictionaries](#hint-dictionaries)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Extensibility](#extensibility)
  - [Custom Backends](#custom-backends)
  - [Custom Voting Strategies](#custom-voting-strategies)
  - [Language Confusion Resolution](#language-confusion-resolution)
  - [Code-Switching Detection](#code-switching-detection)
- [Benchmarks](#benchmarks)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)

---

## The Problem

Traditional language detectors are trained on well-formed sentences and documents. They fail on the kind of text you see in real conversations:

```
"Bonjour!"       -> French (correct)
"Comment ca va?" -> French (correct)
"Bien"           -> Spanish? French? German? (WRONG)
"ok"             -> English? Universal? (AMBIGUOUS)
"thx"            -> Unknown (FAIL)
```

**Why this happens:**
- Short text has low statistical signal
- Words like "ok", "taxi", "pizza" exist in many languages
- Chat slang ("thx", "mdr", "jaja") isn't in training data
- No context from surrounding messages

**FastLangML solves this** by:
1. Tracking conversation context to disambiguate short messages
2. Using hint dictionaries for slang and common words
3. Combining multiple detection backends for robustness
4. Returning "unknown" (`und`) when uncertain instead of wrong guesses

---

## Installation

```bash
# Full installation with all backends (recommended)
pip install fastlangml[all]

# Minimal installation (fasttext only)
pip install fastlangml[fasttext]

# Pick specific backends
pip install fastlangml[fasttext,lingua]
pip install fastlangml[langdetect]
```

**Available backends:**

| Backend | Languages | Speed | Accuracy | Install Extra |
|---------|-----------|-------|----------|---------------|
| fasttext | 176 | Fast | High | `[fasttext]` |
| fastlangid | 177 | Fast | High (CJK) | `[fastlangid]` |
| lingua | 75 | Medium | Very High | `[lingua]` |
| langdetect | 55 | Fast | Medium | `[langdetect]` |
| pycld3 | 107 | Very Fast | Medium | `[pycld3]` |
| langid | 97 | Fast | Medium | `[langid]` |

> **Note:** `fastlangid` provides improved accuracy for Japanese, Korean, Chinese, and Cantonese detection.

**Optional features:**

| Feature | Description | Install Extra |
|---------|-------------|---------------|
| spaCy NER | Named entity recognition for proper noun filtering | `[spacy]` |
| fuzzy | Fuzzy matching for hint dictionaries | `[fuzzy]` |

---

## Quick Start

### Basic Detection

```python
from fastlangml import detect

# Simple detection
result = detect("Hello, how are you?")
print(result.lang)        # "en"
print(result.confidence)  # 0.95
print(result.reliable)    # True
```

### Context-Aware Detection

The key feature of FastLangML is **automatic context tracking**. When you pass a `ConversationContext`, the library:
1. Remembers the last N detected languages
2. Uses this history to resolve ambiguous messages
3. Auto-updates the context after each detection

```python
from fastlangml import detect, ConversationContext

# Create a context to track the conversation
context = ConversationContext()

# French conversation
detect("Bonjour!", context=context).lang        # "fr" (clear)
detect("Comment ca va?", context=context).lang  # "fr" (clear)
detect("Bien", context=context).lang            # "fr" <- context helps!
detect("ok", context=context).lang              # "fr" <- continues French

# The context tracks that this is a French conversation,
# so ambiguous words resolve to French
```

**How context resolution works:**

| Message | Without Context | With Context (after French turns) |
|---------|-----------------|-----------------------------------|
| "ok" | Ambiguous | French (conversation language) |
| "Bien" | Spanish/French/German? | French (matches history) |
| "Si" | Spanish/Italian? | Italian (if conversation was Italian) |
| "Gracias" | Spanish | Spanish (high confidence, ignores context) |

---

## Core Concepts

### Context-Aware Detection

**What is conversation context?**

`ConversationContext` maintains a sliding window of recent language detections. It tracks:
- The detected language of each message
- The confidence score of each detection
- A weighted history favoring recent messages (decay factor)

**Configuration options:**

```python
context = ConversationContext(
    max_turns=2,       # Remember last 2 messages (default)
    decay_factor=0.8,  # Weight recent messages higher
)
```

**When context helps:**
- Ambiguous short messages ("ok", "yes", "no")
- Words shared across languages ("taxi", "hotel", "pizza")
- Mixed script input (romanized non-Latin languages)

**When context doesn't help:**
- Clear language switches (user explicitly changes language)
- High-confidence detections (context is ignored)

**Example: Customer service routing**

```python
from fastlangml import detect, ConversationContext

def route_message(messages: list[str]) -> str:
    """Route conversation to correct language queue."""
    context = ConversationContext()

    for msg in messages:
        result = detect(msg, context=context)

    # Return dominant language of conversation
    return context.dominant_language or "en"

# French customer conversation
messages = ["Bonjour", "J'ai un probleme", "ok", "merci"]
queue = route_message(messages)  # Returns "fr"
```

---

### Multi-Backend Ensemble

FastLangML can combine multiple language detection backends for better accuracy. Each backend has different strengths:

| Backend | Best For | Weaknesses |
|---------|----------|------------|
| fasttext | Speed, many languages | Less accurate on short text |
| lingua | Accuracy, short text | Slower, fewer languages |
| langdetect | General purpose | Non-deterministic by default |
| pycld3 | Speed, CLD3 compatibility | Lower accuracy |

**How ensemble works:**

1. Text is sent to all configured backends
2. Each backend returns a language prediction with confidence
3. A voting strategy combines the predictions
4. Final result is the language with highest combined score

```python
from fastlangml import FastLangDetector, DetectionConfig

# Configure ensemble with 3 backends
detector = FastLangDetector(
    config=DetectionConfig(
        backends=["fasttext", "lingua", "langdetect"],
        backend_weights={
            "fasttext": 0.5,   # Trust fasttext most
            "lingua": 0.35,    # Lingua for accuracy
            "langdetect": 0.15 # Langdetect as tiebreaker
        },
    )
)

result = detector.detect("Ciao, come stai?")
print(result.lang)    # "it"
print(result.backend) # "ensemble"
```

---

### Voting Strategies

Voting strategies determine how to combine predictions from multiple backends.

**Available strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `weighted` | Weighted average of confidence scores | Production (default) |
| `hard` | Majority vote (each backend = 1 vote) | Equal-trust backends |
| `soft` | Average of all probabilities | Well-calibrated backends |
| `consensus` | Require N backends to agree | High-certainty requirements |

**Weighted Voting (Default)**

Multiplies each backend's confidence by its weight, then picks the language with highest weighted score.

```python
from fastlangml import FastLangDetector, DetectionConfig

detector = FastLangDetector(
    config=DetectionConfig(
        voting_strategy="weighted",
        backend_weights={
            "fasttext": 0.6,
            "lingua": 0.4,
        },
    )
)
```

**Hard Voting**

Each backend gets one vote. Ties broken by confidence.

```python
detector = FastLangDetector(
    config=DetectionConfig(voting_strategy="hard")
)
```

**Consensus Voting**

Only returns a result if at least N backends agree. Useful when you need high certainty.

```python
from fastlangml import ConsensusVoting, FastLangDetector, DetectionConfig

detector = FastLangDetector(
    config=DetectionConfig(
        custom_voting=ConsensusVoting(min_agreement=2)
    )
)
```

---

### Hint Dictionaries

Hints are word-to-language mappings that override backend detection. Essential for:
- Chat slang ("thx" -> English, "mdr" -> French)
- Company-specific terms
- Ambiguous words you want to force

**Built-in hints for short text:**

```python
from fastlangml import FastLangDetector, HintDictionary

# Load default hints for chat/SMS
hints = HintDictionary.default_short_words()

detector = FastLangDetector(hints=hints)
detector.detect("thx").lang   # "en" (thanks)
detector.detect("mdr").lang   # "fr" (mort de rire = LOL)
detector.detect("jaja").lang  # "es" (Spanish laugh)
```

**Adding custom hints:**

```python
from fastlangml import FastLangDetector

detector = FastLangDetector()

# Add hints for your domain
detector.add_hint("asap", "en")
detector.add_hint("btw", "en")
detector.add_hint("stp", "fr")  # s'il te plait

# Hints override backend detection
detector.detect("asap").lang  # "en"
```

**Hint priority:**

Hints are checked before backend detection. If a hint matches:
1. The hint's language gets a confidence boost
2. For very short text (<=5 chars), hints dominate the result
3. For longer text, hints are weighted with backend predictions

---

## API Reference

### detect()

The main function for language detection.

```python
def detect(
    text: str,
    context: ConversationContext | None = None,
    mode: str = "default",
    auto_update: bool = True,
) -> DetectionResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Text to detect |
| `context` | `ConversationContext` | `None` | Conversation context for multi-turn |
| `mode` | `str` | `"default"` | Detection mode: `"short"`, `"default"`, `"long"` |
| `auto_update` | `bool` | `True` | Automatically add result to context |

**Returns:** `DetectionResult`

### DetectionResult

```python
@dataclass
class DetectionResult:
    lang: str           # ISO 639-1 code ("en", "fr", "und")
    confidence: float   # 0.0 to 1.0
    reliable: bool      # True if confidence >= threshold
    reason: str | None  # Why "und" was returned (if applicable)
    script: str | None  # Detected script ("latin", "cyrillic", etc.)
    backend: str        # Which backend or "ensemble"
    candidates: list    # Top-k alternatives
    meta: dict          # Timing and debug info
```

### ConversationContext

```python
context = ConversationContext(
    max_turns: int = 2,      # Max messages to remember
    decay_factor: float = 0.8, # Recency weight (0.0-1.0)
)

# Properties
context.dominant_language    # Most common language in history
context.language_distribution # {lang: weighted_count}
context.last_turn           # Most recent turn

# Methods
context.get_context_boost(lang)  # Get boost score for a language
context.get_language_streak()    # (lang, streak_count)
context.clear()                  # Reset context
```

### FastLangDetector

```python
detector = FastLangDetector(
    config=DetectionConfig(...),
    hints=HintDictionary(),
)

# Methods
detector.detect(text, context=None, mode="default")
detector.detect_batch(texts, mode="default")
detector.add_hint(word, lang)
detector.remove_hint(word)
detector.set_languages(["en", "fr", "es"])  # Restrict output
detector.available_backends()
```

---

## Configuration

### DetectionConfig

```python
from fastlangml import DetectionConfig

config = DetectionConfig(
    # Backend configuration
    backends=["fasttext", "lingua"],
    backend_weights={"fasttext": 0.6, "lingua": 0.4},

    # Voting
    voting_strategy="weighted",  # or "hard", "soft", "consensus"
    custom_voting=None,          # VotingStrategy instance

    # Thresholds
    thresholds={
        "short": 0.5,   # Confidence threshold for short mode
        "default": 0.7,
        "long": 0.8,
    },
    min_text_length=1,

    # Features
    filter_proper_nouns=False,  # Enable heuristic-based proper noun filtering
    use_script_filter=True,

    # spaCy NER for proper noun filtering (requires pip install fastlangml[spacy])
    # filter = ProperNounFilter(use_spacy=True)
    # Filters: PERSON, ORG, GPE, LOC, FAC, NORP entities

    # Weights
    hint_weight=1.5,
    context_weight=0.3,
)
```

---

## Extensibility

### Custom Backends

Create your own detection backend using the `@backend` decorator:

```python
from fastlangml import backend, Backend
from fastlangml.backends.base import DetectionResult

@backend("my_detector", reliability=4)  # reliability: 1-5
class MyBackend(Backend):
    """Custom language detection backend."""

    @property
    def name(self) -> str:
        return "my_detector"

    @property
    def is_available(self) -> bool:
        # Check if required dependencies are installed
        return True

    def detect(self, text: str) -> DetectionResult:
        # Your detection logic here
        lang = "en"  # Replace with actual detection
        confidence = 0.95
        return DetectionResult(self.name, lang, confidence)

    def supported_languages(self) -> set[str]:
        return {"en", "fr", "de", "es"}
```

**Using the custom backend:**

```python
from fastlangml import FastLangDetector, DetectionConfig

detector = FastLangDetector(
    config=DetectionConfig(
        backends=["my_detector", "fasttext"],
    )
)
```

**Programmatic registration:**

```python
from fastlangml import register_backend, unregister_backend, list_registered_backends

# Register
register_backend("my_backend", MyBackend, reliability=4)

# List all registered
print(list_registered_backends())  # ["my_backend"]

# Unregister
unregister_backend("my_backend")
```

---

### Custom Voting Strategies

Implement your own voting logic by extending `VotingStrategy`:

```python
from fastlangml import VotingStrategy, FastLangDetector, DetectionConfig

class ConfidenceOnlyVoting(VotingStrategy):
    """Pick the language with highest individual confidence."""

    def vote(
        self,
        results: list,
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not results:
            return {}

        # Find max confidence per language
        scores = {}
        for r in results:
            lang = r.language
            if lang not in scores or r.confidence > scores[lang]:
                scores[lang] = r.confidence

        return scores

# Use custom voting
detector = FastLangDetector(
    config=DetectionConfig(
        custom_voting=ConfidenceOnlyVoting()
    )
)
```

---

### Language Confusion Resolution

FastLangML handles commonly confused language pairs with specialized logic:

**Supported confused pairs:**
- Spanish / Portuguese
- Norwegian / Danish / Swedish
- Czech / Slovak
- Croatian / Serbian / Bosnian
- Indonesian / Malay
- Russian / Ukrainian / Belarusian
- Hindi / Urdu

```python
from fastlangml import ConfusionResolver, LanguageSimilarity

# Resolve ambiguity between similar languages
resolver = ConfusionResolver()

# Check if languages are a known confused pair
pair = resolver.get_confused_pair({"es", "pt"})  # frozenset({"es", "pt"})

# Adjust scores based on discriminating features
scores = {"es": 0.45, "pt": 0.42}
adjusted = resolver.resolve("Eu tenho um problema", scores)
# Portuguese boosted due to "tenho" (have)

# Get discriminating features
es_features, pt_features = resolver.get_discriminating_features("es", "pt")
# es_features: ["pero", "cuando", "donde", ...]
# pt_features: ["mas", "quando", "onde", ...]

# Check language relationships
sim = LanguageSimilarity()
sim.are_related("es", "pt")  # True (Romance family)
sim.are_related("en", "zh")  # False (different families)
sim.get_related_languages("es")  # {"pt", "fr", "it", "ro", "ca", "gl"}
```

---

### Code-Switching Detection

Detect mixed-language messages (Spanglish, Franglais, Hinglish, etc.):

```python
from fastlangml import CodeSwitchDetector

detector = CodeSwitchDetector()

# Detect code-switching
result = detector.detect("That's muy importante for the proyecto")

result.is_mixed            # True
result.primary_language    # "en"
result.secondary_languages # ["es"]
result.languages           # ["en", "es"]
result.language_distribution  # {"en": 0.6, "es": 0.4}

# Get language spans
for span in result.spans:
    print(f"{span.text}: {span.language} ({span.confidence:.2f})")
# "That's": en (0.85)
# "muy": es (0.92)
# "importante": es (0.95)
# "for": en (0.88)
# "the": en (0.90)
# "proyecto": es (0.94)

# Quick check
detector.is_code_switched("Hello world")  # False
detector.is_code_switched("Hola, how are you?")  # True

# Pattern-based detection
from fastlangml import detect_code_switching_pattern

pattern = detect_code_switching_pattern("That's muy bueno")
# ("en", "es") - matches Spanglish pattern
```

**Supported code-switching patterns:**
- Spanglish (English + Spanish)
- Franglais (English + French)
- Hinglish (English + Hindi)
- Denglish (German + English)

---

## Benchmarks

### Accuracy on Short Text

Tested on 1,000 short messages (2-10 words) from multilingual chat datasets:

| Configuration | Accuracy | Avg Latency |
|---------------|----------|-------------|
| fasttext only | 82.3% | 0.8ms |
| lingua only | 89.1% | 4.2ms |
| ensemble (ft+lingua) | 91.4% | 3.1ms |
| ensemble + context | 94.7% | 3.3ms |
| ensemble + context + hints | 96.2% | 3.4ms |

### Latency by Backend

Single message detection latency (median, P99):

| Backend | Median | P99 |
|---------|--------|-----|
| fasttext | 0.5ms | 2.1ms |
| langdetect | 1.2ms | 5.8ms |
| lingua | 3.8ms | 12.4ms |
| pycld3 | 0.3ms | 1.5ms |
| ensemble (3 backends) | 2.4ms | 8.2ms |

### Throughput

Batch detection throughput (messages/second):

| Configuration | Single Thread | 4 Threads |
|---------------|---------------|-----------|
| fasttext | 12,500 | 42,000 |
| ensemble (2 backends) | 4,200 | 15,800 |
| ensemble (3 backends) | 2,100 | 8,400 |

### Running Benchmarks

```bash
# Built-in benchmark command
fastlangml bench --n-samples 1000 --languages en,fr,es,de

# With specific dataset
fastlangml bench --dataset wili --n-samples 500
```

---

## Best Practices

### 1. Use Context for Chat/Messaging

Always pass a `ConversationContext` when detecting messages in a conversation:

```python
# Good: Context-aware
context = ConversationContext()
for msg in conversation:
    result = detect(msg, context=context)

# Bad: Stateless detection
for msg in conversation:
    result = detect(msg)  # Loses valuable context
```

### 2. Choose the Right Mode

- **short**: For SMS, chat messages (< 50 chars). Lower confidence threshold.
- **default**: General purpose. Balanced threshold.
- **long**: For paragraphs/documents. Higher confidence threshold.

```python
detect("ok", mode="short")         # More lenient
detect("Hello world", mode="default")
detect(long_paragraph, mode="long") # More strict
```

### 3. Add Domain-Specific Hints

If your users use specific slang or terms, add hints:

```python
detector.add_hint("gg", "en")     # Gaming
detector.add_hint("lol", "en")
detector.add_hint("mdr", "fr")    # French LOL
detector.add_hint("kek", "en")    # Gaming laugh
```

### 4. Restrict Languages When Known

If you know the possible languages (e.g., a bilingual support queue), restrict the output:

```python
# Only consider English and Spanish
detector.set_languages(["en", "es"])
```

### 5. Handle "und" (Unknown)

When FastLangML is uncertain, it returns `und` instead of guessing wrong:

```python
result = detect("ok")
if result.lang == "und":
    # Fallback to default or ask user
    lang = result.candidates[0].lang if result.candidates else "en"
    print(f"Low confidence: {result.reason}")
```

### 6. Use Ensemble for Production

Single backends have blind spots. Ensemble improves reliability:

```python
# Development: Fast single backend
detector = FastLangDetector(
    config=DetectionConfig(backends=["fasttext"])
)

# Production: Reliable ensemble
detector = FastLangDetector(
    config=DetectionConfig(
        backends=["fasttext", "lingua"],
        voting_strategy="weighted",
    )
)
```

### 7. Batch Detection for Throughput

When processing many messages, use batch detection:

```python
# Good: Batch processing (parallelized)
results = detector.detect_batch(messages, mode="short")

# Bad: Sequential detection
results = [detector.detect(msg) for msg in messages]  # Slower
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Start

```bash
# Clone and setup
git clone https://github.com/pnrajan/fastlangml.git
cd fastlangml
pip install -e ".[dev,all]"

# Development commands
make test       # Run tests
make lint       # Check code style
make fix        # Auto-fix linting issues
make typecheck  # Run type checker
make check      # Run all checks
```

### What We're Looking For

- Bug fixes with test cases
- New detection backends
- Performance improvements
- Documentation improvements
- New voting strategies

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Project architecture overview
- Testing guidelines
- Code style requirements
- Commit message conventions
- Pull request process

---

## License

MIT License - see [LICENSE](LICENSE) for details.
