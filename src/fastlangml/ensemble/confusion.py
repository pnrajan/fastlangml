"""Language confusion matrix and similar language pair handling.

Provides specialized logic for handling commonly confused language pairs
like Spanish/Portuguese, Norwegian/Danish/Swedish, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Language pairs that are commonly confused by detectors
# Format: (lang1, lang2): discriminating_features
CONFUSED_PAIRS: dict[frozenset[str], dict[str, list[str]]] = {
    # Spanish vs Portuguese
    frozenset({"es", "pt"}): {
        "es": [
            "pero",
            "porque",
            "cuando",
            "donde",
            "como",
            "este",
            "esta",
            "tiene",
            "puede",
            "hace",
            "muy",
            "siempre",
            "nunca",
            "también",
            "ahora",
            "después",
            "antes",
            "sobre",
            "bajo",
            "entre",
            "ñ",  # Spanish has ñ
            "ll",  # Spanish double-l
        ],
        "pt": [
            "mas",
            "porque",
            "quando",
            "onde",
            "como",
            "este",
            "esta",
            "tem",
            "pode",
            "faz",
            "muito",
            "sempre",
            "nunca",
            "também",
            "agora",
            "depois",
            "antes",
            "sobre",
            "baixo",
            "entre",
            "ão",
            "ões",
            "ção",  # Portuguese endings
            "lh",
            "nh",  # Portuguese digraphs
            "você",
            "vocês",
            "é",
            "são",
        ],
    },
    # Norwegian vs Danish vs Swedish (Scandinavian)
    frozenset({"no", "da", "sv"}): {
        "no": [
            "jeg",
            "ikke",
            "det",
            "og",
            "på",
            "er",
            "en",
            "av",
            "å",
            "være",
            "har",
            "kan",
            "vil",
            "skal",
            "må",
            "hva",
            "hvor",
            "når",
            "hvorfor",
            "hvem",
            "også",
            "bare",
            "aldri",
            "alltid",
        ],
        "da": [
            "jeg",
            "ikke",
            "det",
            "og",
            "på",
            "er",
            "en",
            "af",
            "at",
            "være",
            "har",
            "kan",
            "vil",
            "skal",
            "må",
            "hvad",
            "hvor",
            "hvornår",
            "hvorfor",
            "hvem",
            "også",
            "bare",
            "aldrig",
            "altid",
            "ø",
            "æ",  # Danish characters
        ],
        "sv": [
            "jag",
            "inte",
            "det",
            "och",
            "på",
            "är",
            "en",
            "av",
            "att",
            "vara",
            "har",
            "kan",
            "vill",
            "ska",
            "måste",
            "vad",
            "var",
            "när",
            "varför",
            "vem",
            "också",
            "bara",
            "aldrig",
            "alltid",
            "ä",
            "ö",  # Swedish characters
        ],
    },
    # Czech vs Slovak
    frozenset({"cs", "sk"}): {
        "cs": [
            "být",
            "mít",
            "dělat",
            "říct",
            "jít",
            "vědět",
            "chtít",
            "já",
            "ty",
            "on",
            "ona",
            "my",
            "vy",
            "oni",
            "ano",
            "ne",
            "proč",
            "jak",
            "kdy",
            "kde",
            "ř",  # Czech has ř
            "ě",
            "ů",
        ],
        "sk": [
            "byť",
            "mať",
            "robiť",
            "povedať",
            "ísť",
            "vedieť",
            "chcieť",
            "ja",
            "ty",
            "on",
            "ona",
            "my",
            "vy",
            "oni",
            "áno",
            "nie",
            "prečo",
            "ako",
            "kedy",
            "kde",
            "ä",
            "ô",
            "ĺ",
            "ŕ",  # Slovak characters
        ],
    },
    # Croatian vs Serbian vs Bosnian
    frozenset({"hr", "sr", "bs"}): {
        "hr": [
            "tko",
            "što",
            "zašto",
            "gdje",
            "kako",
            "kada",
            "samo",
            "već",
            "još",
            "sada",
            "ovdje",
            "tamo",
            "trebati",
            "moći",
            "htjeti",
            "znati",
        ],
        "sr": [
            "ко",
            "шта",
            "зашто",
            "где",
            "како",
            "када",  # Cyrillic
            "ko",
            "šta",
            "zašto",
            "gde",
            "kako",
            "kada",  # Latin
            "само",
            "већ",
            "још",
            "сада",
            "овде",
            "тамо",
        ],
        "bs": [
            "ko",
            "šta",
            "zašto",
            "gdje",
            "kako",
            "kada",
            "samo",
            "već",
            "još",
            "sada",
            "ovdje",
            "tamo",
        ],
    },
    # Indonesian vs Malay
    frozenset({"id", "ms"}): {
        "id": [
            "tidak",
            "bukan",
            "dengan",
            "yang",
            "untuk",
            "dari",
            "adalah",
            "akan",
            "telah",
            "sedang",
            "sudah",
            "saya",
            "anda",
            "mereka",
            "kami",
            "kita",
            "apa",
            "siapa",
            "mengapa",
            "bagaimana",
            "dimana",
        ],
        "ms": [
            "tidak",
            "bukan",
            "dengan",
            "yang",
            "untuk",
            "dari",
            "ialah",
            "akan",
            "telah",
            "sedang",
            "sudah",
            "saya",
            "anda",
            "mereka",
            "kami",
            "kita",
            "apa",
            "siapa",
            "kenapa",
            "macam mana",
            "di mana",
        ],
    },
    # Russian vs Ukrainian vs Belarusian
    frozenset({"ru", "uk", "be"}): {
        "ru": [
            "что",
            "как",
            "это",
            "он",
            "она",
            "они",
            "мы",
            "вы",
            "быть",
            "есть",
            "был",
            "была",
            "были",
            "будет",
            "и",
            "в",
            "не",
            "на",
            "с",
            "по",
            "для",
            "от",
        ],
        "uk": [
            "що",
            "як",
            "це",
            "він",
            "вона",
            "вони",
            "ми",
            "ви",
            "бути",
            "є",
            "був",
            "була",
            "були",
            "буде",
            "і",
            "в",
            "не",
            "на",
            "з",
            "по",
            "для",
            "від",
            "ї",
            "і",
            "є",  # Ukrainian characters
        ],
        "be": [
            "што",
            "як",
            "гэта",
            "ён",
            "яна",
            "яны",
            "мы",
            "вы",
            "быць",
            "ёсць",
            "быў",
            "была",
            "былі",
            "будзе",
            "і",
            "у",
            "не",
            "на",
            "з",
            "па",
            "для",
            "ад",
            "ў",
            "і",  # Belarusian characters
        ],
    },
    # Hindi vs Urdu (written differently but spoken similarly)
    frozenset({"hi", "ur"}): {
        "hi": [
            "है",
            "हैं",
            "था",
            "थे",
            "और",
            "का",
            "की",
            "के",
            "को",
            "में",
            "से",
            "पर",
            "यह",
            "वह",
            "क्या",
        ],
        "ur": [
            "ہے",
            "ہیں",
            "تھا",
            "تھے",
            "اور",
            "کا",
            "کی",
            "کے",
            "کو",
            "میں",
            "سے",
            "پر",
            "یہ",
            "وہ",
            "کیا",
        ],
    },
}


@dataclass
class ConfusionResolver:
    """Resolves ambiguity between commonly confused language pairs.

    Uses discriminating features (words, characters, patterns) to distinguish
    between similar languages when confidence is low or multiple backends disagree.

    Example:
        >>> resolver = ConfusionResolver()
        >>> resolver.resolve("Eu tenho um problema", {"es": 0.45, "pt": 0.42})
        {"es": 0.35, "pt": 0.62}  # Portuguese boosted due to "tenho"
    """

    min_confidence_gap: float = 0.15
    """Minimum gap before applying confusion resolution."""

    boost_factor: float = 0.2
    """Amount to boost/penalize based on discriminating features."""

    def get_confused_pair(self, langs: set[str]) -> frozenset[str] | None:
        """Check if the given languages are a known confused pair."""
        for pair in CONFUSED_PAIRS:
            if langs <= pair or (len(langs) == 2 and langs == pair):
                return pair
        return None

    def resolve(
        self,
        text: str,
        scores: dict[str, float],
        top_n: int = 2,
    ) -> dict[str, float]:
        """Resolve confusion between similar languages.

        Args:
            text: Input text to analyze
            scores: Current language scores from detection
            top_n: Number of top languages to consider

        Returns:
            Adjusted scores with confusion resolution applied
        """
        if not scores or len(scores) < 2:
            return scores

        # Get top N languages
        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_langs = {lang for lang, _ in sorted_langs[:top_n]}

        # Check if top languages are a confused pair
        confused_pair = self.get_confused_pair(top_langs)
        if not confused_pair:
            return scores

        # Check if scores are close enough to warrant resolution
        if len(sorted_langs) >= 2:
            gap = sorted_langs[0][1] - sorted_langs[1][1]
            if gap > self.min_confidence_gap:
                return scores  # Clear winner, no need to resolve

        # Apply discriminating features
        text_lower = text.lower()
        feature_scores: dict[str, float] = {lang: 0.0 for lang in confused_pair}

        pair_features = CONFUSED_PAIRS.get(confused_pair, {})
        for lang, features in pair_features.items():
            if lang not in scores:
                continue
            for feature in features:
                if feature in text_lower:
                    feature_scores[lang] += 1

        # Normalize and apply boost
        total_features = sum(feature_scores.values())
        if total_features > 0:
            adjusted = dict(scores)
            for lang in confused_pair:
                if lang in adjusted:
                    boost = (feature_scores[lang] / total_features) * self.boost_factor
                    adjusted[lang] = min(adjusted[lang] + boost, 1.0)
            return adjusted

        return scores

    def get_discriminating_features(
        self, lang1: str, lang2: str
    ) -> tuple[list[str], list[str]] | None:
        """Get discriminating features between two languages.

        Returns:
            Tuple of (lang1_features, lang2_features) or None if not a known pair
        """
        pair = frozenset({lang1, lang2})
        if pair not in CONFUSED_PAIRS:
            return None

        features = CONFUSED_PAIRS[pair]
        return features.get(lang1, []), features.get(lang2, [])


@dataclass
class LanguageSimilarity:
    """Tracks similarity relationships between languages.

    Used to understand when detection ambiguity is expected vs. surprising.
    """

    # Language families - languages within a family are more likely to be confused
    FAMILIES: dict[str, set[str]] = field(
        default_factory=lambda: {
            "romance": {"es", "pt", "fr", "it", "ro", "ca", "gl"},
            "germanic": {"en", "de", "nl", "sv", "da", "no", "is"},
            "slavic": {"ru", "uk", "be", "pl", "cs", "sk", "hr", "sr", "bs", "bg", "sl"},
            "scandinavian": {"sv", "da", "no", "is"},
            "south_slavic": {"hr", "sr", "bs", "bg", "sl", "mk"},
            "east_slavic": {"ru", "uk", "be"},
            "west_slavic": {"pl", "cs", "sk"},
            "cjk": {"zh", "ja", "ko"},
            "semitic": {"ar", "he"},
            "indic": {"hi", "ur", "bn", "pa", "mr", "gu"},
            "austronesian": {"id", "ms", "tl"},
        }
    )

    def get_family(self, lang: str) -> str | None:
        """Get the language family for a given language."""
        for family, langs in self.FAMILIES.items():
            if lang in langs:
                return family
        return None

    def are_related(self, lang1: str, lang2: str) -> bool:
        """Check if two languages are in the same family."""
        return any(lang1 in langs and lang2 in langs for langs in self.FAMILIES.values())

    def get_related_languages(self, lang: str) -> set[str]:
        """Get all languages related to the given one."""
        related = set()
        for langs in self.FAMILIES.values():
            if lang in langs:
                related.update(langs)
        related.discard(lang)
        return related

    def similarity_score(self, lang1: str, lang2: str) -> float:
        """Get a similarity score between two languages (0.0 to 1.0)."""
        if lang1 == lang2:
            return 1.0

        # Check for known confused pairs (highest similarity)
        pair = frozenset({lang1, lang2})
        if pair in CONFUSED_PAIRS:
            return 0.9

        # Check for same family
        if self.are_related(lang1, lang2):
            return 0.6

        # Different families
        return 0.0


__all__ = [
    "CONFUSED_PAIRS",
    "ConfusionResolver",
    "LanguageSimilarity",
]
