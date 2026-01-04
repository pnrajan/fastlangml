"""Name filtering for language detection.

Filters out person/company names before language detection to avoid
names being misdetected as a language.
"""

from __future__ import annotations

import re


def filter_names(text: str, use_probablepeople: bool = True) -> tuple[str, list[str]]:
    """
    Filter out detected names from text.

    Args:
        text: Input text
        use_probablepeople: Use probablepeople for name detection

    Returns:
        Tuple of (filtered_text, list of detected names)
    """
    if not text or len(text) < 2:
        return text, []

    detected_names = []
    filtered = text

    if use_probablepeople:
        try:
            import probablepeople as pp

            # Try to parse as name
            try:
                parsed, name_type = pp.tag(text)
                if name_type == 'Person':
                    # Extract name parts
                    name_parts = []
                    for key, value in parsed.items():
                        if key in ('GivenName', 'Surname', 'MiddleName',
                                   'FirstInitial', 'LastInitial', 'Nickname'):
                            name_parts.append(value)
                            detected_names.append(value)

                    # Remove name parts from text
                    for part in name_parts:
                        pattern = r'\b' + re.escape(part) + r'\b'
                        filtered = re.sub(pattern, '', filtered, flags=re.IGNORECASE)
                    filtered = ' '.join(filtered.split())

            except pp.RepeatedLabelError:
                pass  # Not a parseable name

        except ImportError:
            pass  # probablepeople not installed

    return filtered, detected_names


def is_likely_name(text: str) -> bool:
    """
    Quick heuristic to check if text is likely a person name.

    Args:
        text: Input text

    Returns:
        True if text looks like a name
    """
    # Simple heuristics
    words = text.split()

    if len(words) < 1 or len(words) > 5:
        return False

    # All words start with capital
    if all(w[0].isupper() for w in words if w):
        # No common function words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                        'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        if not any(w.lower() in common_words for w in words):
            return True

    return False
