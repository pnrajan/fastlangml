# Changelog

All notable changes to FastLangML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-03

### Initial Release

FastLangML is a high-accuracy language detection library optimized for short, conversational text like chat messages, SMS, and customer service interactions.

### Features

#### Multi-Backend Ensemble
- **FastText** - Fast detection supporting 176 languages
- **Lingua** - High accuracy for short text, 75 languages
- **langdetect** - Google's language detection, 55 languages
- **pyCLD3** - Compact Language Detector v3, 107 languages
- **langid** - Lightweight language identification
- Configurable backend weights and voting strategies

#### Voting Strategies
- **Weighted Voting** - Weighted average of confidence scores (default)
- **Hard Voting** - Majority vote (one vote per backend)
- **Soft Voting** - Average of all probabilities
- **Consensus Voting** - Require N backends to agree
- Custom voting strategies via `VotingStrategy` base class

#### Context-Aware Detection
- `ConversationContext` tracks conversation history
- Resolves ambiguous short messages ("ok", "si", "bien") using context
- Configurable max turns and decay factor
- Auto-updates context after each detection

#### Code-Switching Detection
- Detects mixed-language messages (Spanglish, Franglais, Hinglish, Denglish)
- `CodeSwitchDetector` with word-level and segment-level detection
- Returns language spans with confidence scores
- Pattern-based detection for common code-switching patterns

#### Language Confusion Resolution
- Handles commonly confused language pairs:
  - Spanish / Portuguese
  - Norwegian / Danish / Swedish
  - Czech / Slovak
  - Croatian / Serbian / Bosnian
  - Indonesian / Malay
  - Russian / Ukrainian / Belarusian
  - Hindi / Urdu
- `ConfusionResolver` adjusts scores based on discriminating features
- `LanguageSimilarity` for language family relationships

#### Hint Dictionaries
- Built-in hints for chat slang and abbreviations (25+ languages)
- Supports: "thx", "mdr", "jaja", "kk", "rsrs", and 500+ terms
- Runtime hint addition via `add_hint(word, lang)`
- File persistence (JSON/TOML) for custom dictionaries

#### Preprocessing
- **Proper noun filtering** - Remove/mask names before detection
- **Script detection** - Identify Latin, Cyrillic, Arabic, CJK, etc.
- **Script-based filtering** - Narrow candidates by script

#### CLI Tool
- `fastlangml "text"` - Detect language
- `fastlangml --mode short "ok"` - Short text mode
- `fastlangml batch file.txt` - Batch processing
- `fastlangml bench` - Run benchmarks
- JSON output format

#### Extensibility
- `@backend` decorator for custom backends
- `register_backend()` / `unregister_backend()` functions
- `VotingStrategy` base class for custom voting
- Plugin architecture for adding new detection engines

### API

```python
from fastlangml import detect, ConversationContext

# Simple detection
result = detect("Hello world")
print(result.lang)        # "en"
print(result.confidence)  # 0.95

# Context-aware detection
context = ConversationContext()
detect("Bonjour!", context=context)      # "fr"
detect("Comment ça va?", context=context) # "fr"
detect("Bien", context=context)           # "fr" (context helps!)

# Code-switching detection
from fastlangml import CodeSwitchDetector
detector = CodeSwitchDetector()
result = detector.detect("That's muy importante")
print(result.is_mixed)           # True
print(result.primary_language)   # "en"
print(result.secondary_languages) # ["es"]
```

### Supported Languages

170+ languages via multi-backend ensemble, including:
- All major world languages (English, Spanish, Chinese, Arabic, Hindi, etc.)
- European languages (French, German, Italian, Portuguese, Dutch, etc.)
- Asian languages (Japanese, Korean, Vietnamese, Thai, Indonesian, etc.)
- Slavic languages (Russian, Ukrainian, Polish, Czech, etc.)
- And many more...

### Requirements

- Python 3.10+
- At least one backend installed (fasttext recommended)

### Installation

```bash
# Full installation (recommended)
pip install fastlangml[all]

# Minimal installation
pip install fastlangml[fasttext]

# Specific backends
pip install fastlangml[fasttext,lingua]
```

### Links

- **Repository**: https://github.com/pnrajan/FastLangML
- **Documentation**: https://github.com/pnrajan/FastLangML#readme
- **Issues**: https://github.com/pnrajan/FastLangML/issues

[1.0.0]: https://github.com/pnrajan/FastLangML/releases/tag/v1.0.0
