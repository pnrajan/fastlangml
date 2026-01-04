"""Persistence layer for hint dictionaries.

Supports loading and saving hints to TOML and JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastlangml.hints.dictionary import HintDictionary

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]

try:
    import tomli_w

    HAS_TOML_WRITE = True
except ImportError:
    HAS_TOML_WRITE = False


class HintPersistence:
    """
    Persist and load hint dictionaries to/from files.

    Supports:
    - TOML (recommended for config)
    - JSON (universal)

    TOML format:
        [hints]
        bonjour = "fr"
        gracias = "es"
        danke = "de"

    JSON format:
        {"hints": {"bonjour": "fr", "gracias": "es", "danke": "de"}}
    """

    @staticmethod
    def load_toml(path: Path | str) -> HintDictionary:
        """
        Load hints from a TOML file.

        Args:
            path: Path to TOML file

        Returns:
            HintDictionary populated with hints from file
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        hints = HintDictionary()
        for word, lang in data.get("hints", {}).items():
            hints.add(word, lang)

        return hints

    @staticmethod
    def save_toml(hints: HintDictionary, path: Path | str) -> None:
        """
        Save hints to a TOML file.

        Args:
            hints: HintDictionary to save
            path: Path to write TOML file

        Raises:
            ImportError: If tomli_w is not installed
        """
        if not HAS_TOML_WRITE:
            raise ImportError(
                "tomli_w required for TOML writing. Install with: pip install tomli-w"
            )

        path = Path(path)
        data = {"hints": dict(hints.items())}

        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    @staticmethod
    def load_json(path: Path | str) -> HintDictionary:
        """
        Load hints from a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            HintDictionary populated with hints from file
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        hints = HintDictionary()

        # Support both {"hints": {...}} and flat {...} formats
        hint_data = data.get("hints", data)
        for word, lang in hint_data.items():
            hints.add(word, lang)

        return hints

    @staticmethod
    def save_json(hints: HintDictionary, path: Path | str, indent: int = 2) -> None:
        """
        Save hints to a JSON file.

        Args:
            hints: HintDictionary to save
            path: Path to write JSON file
            indent: JSON indentation level
        """
        path = Path(path)
        data = {"hints": dict(hints.items())}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def load(path: Path | str) -> HintDictionary:
        """
        Auto-detect format and load hints.

        Args:
            path: Path to hints file

        Returns:
            HintDictionary populated with hints from file

        Raises:
            ValueError: If file format is not supported
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".toml":
            return HintPersistence.load_toml(path)
        elif suffix == ".json":
            return HintPersistence.load_json(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def save(hints: HintDictionary, path: Path | str) -> None:
        """
        Auto-detect format and save hints.

        Args:
            hints: HintDictionary to save
            path: Path to write file

        Raises:
            ValueError: If file format is not supported
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".toml":
            HintPersistence.save_toml(hints, path)
        elif suffix == ".json":
            HintPersistence.save_json(hints, path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
