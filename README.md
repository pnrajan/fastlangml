# FastLangID

[![PyPI](https://img.shields.io/pypi/v/fastlangid)](https://pypi.org/project/fastlangid/)
[![Python](https://img.shields.io/pypi/pyversions/fastlangid)](https://pypi.org/project/fastlangid/)
[![Tests](https://github.com/pankajrajan/fastlangid/actions/workflows/tests.yml/badge.svg)](https://github.com/pankajrajan/fastlangid/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Language detection built for conversations.** One line to detect, pass context for accuracy.

## The Problem

Traditional language detectors fail on short conversational text:

```
"Bonjour!"       → French ✓
"Comment ça va?" → French ✓
"Bien"           → Spanish? French? German? ✗
```

FastLangID uses conversation context to resolve ambiguity:

```python
from fastlangid import detect, ConversationContext

context = ConversationContext()
detect("Bonjour!", context=context).lang   # "fr"
detect("Bien", context=context).lang       # "fr" ← context helps!
```

## Install

```bash
pip install fastlangid[all]
```

## Quick Start

```python
from fastlangid import detect

detect("Hello world").lang        # "en"
detect("Bonjour le monde").lang   # "fr"
```

## Context-Aware Detection

Context is automatically updated after each detection:

```python
from fastlangid import detect, ConversationContext

context = ConversationContext()

detect("Bonjour!", context=context).lang        # "fr"
detect("Comment ça va?", context=context).lang  # "fr"
detect("Bien", context=context).lang            # "fr" ← ambiguous word resolved!
detect("ok", context=context).lang              # "fr" ← continues French context
```

| Message | Without Context | With Context |
|---------|-----------------|--------------|
| "ok" | Ambiguous | Matches conversation language |
| "Bien" | Spanish/French/German? | French (previous turns were French) |
| "Si" | Spanish/Italian? | Italian (if conversation was Italian) |

## Why Conversations Need This

**Customer Service**: Short messages like "Thanks", "OK", "Sure" are ambiguous without context.

**SMS/Chat**: Character limits mean less signal. Slang like "thx", "mdr", "jaja" needs hints.

**Chatbots**: Must route to correct language model instantly using multi-turn memory.

## API Reference

```python
detect(
    text,
    context=None,        # ConversationContext for multi-turn accuracy
    mode="short",        # "short" | "default" | "long"
    auto_update=True,    # Auto-update context after detection
) -> DetectionResult
```

**DetectionResult fields:** `lang`, `confidence`, `reliable`, `reason`, `script`

## Configuration

```python
from fastlangid import FastLangDetector, DetectionConfig

detector = FastLangDetector(
    config=DetectionConfig(
        backends=["fasttext", "lingua"],
        voting_strategy="weighted",
    )
)
```

## Backends

| Backend | Languages | Install |
|---------|-----------|---------|
| fasttext | 176 | `pip install fastlangid[fasttext]` |
| lingua | 75 | `pip install fastlangid[lingua]` |
| langdetect | 55 | `pip install fastlangid[langdetect]` |

## Extensibility

### Custom Backends

```python
from fastlangid import backend, Backend
from fastlangid.backends.base import DetectionResult

@backend("mybackend", reliability=4)
class MyBackend(Backend):
    @property
    def name(self) -> str:
        return "mybackend"

    @property
    def is_available(self) -> bool:
        return True

    def detect(self, text: str) -> DetectionResult:
        # Your detection logic
        return DetectionResult(self.name, "en", 0.9)

    def supported_languages(self) -> set[str]:
        return {"en", "fr", "de"}
```

### Custom Voting

```python
from fastlangid import VotingStrategy, FastLangDetector, DetectionConfig

class MyVoting(VotingStrategy):
    def vote(self, results, weights=None):
        scores = {}
        for r in results:
            scores[r.language] = scores.get(r.language, 0) + r.confidence
        return scores

detector = FastLangDetector(
    config=DetectionConfig(custom_voting=MyVoting())
)
```

## Hints for Short Text

```python
from fastlangid import FastLangDetector, HintDictionary

hints = HintDictionary.default_short_words()
detector = FastLangDetector(hints=hints)

detector.detect("thx").lang   # "en"
detector.detect("mdr").lang   # "fr" (French LOL)
detector.detect("jaja").lang  # "es" (Spanish laugh)
```

## License

MIT
