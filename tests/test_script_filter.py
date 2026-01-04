"""Tests for script-based filtering."""


from fastlangml.preprocessing.script_filter import (
    Script,
    ScriptFilter,
    detect_script,
)


class TestDetectScript:
    """Tests for detect_script function."""

    def test_cyrillic(self):
        """Test Cyrillic script detection."""
        script, proportion = detect_script("Привет мир")
        assert script == Script.CYRILLIC
        assert proportion > 0.9

    def test_arabic(self):
        """Test Arabic script detection."""
        script, proportion = detect_script("مرحبا بالعالم")
        assert script == Script.ARABIC
        assert proportion > 0.9

    def test_hebrew(self):
        """Test Hebrew script detection."""
        script, proportion = detect_script("שלום עולם")
        assert script == Script.HEBREW
        assert proportion > 0.9

    def test_greek(self):
        """Test Greek script detection."""
        script, proportion = detect_script("Γεια σου κόσμε")
        assert script == Script.GREEK
        assert proportion > 0.9

    def test_latin(self):
        """Test Latin script detection."""
        script, proportion = detect_script("Hello world")
        assert script == Script.LATIN
        assert proportion > 0.9

    def test_devanagari(self):
        """Test Devanagari script detection."""
        script, proportion = detect_script("नमस्ते दुनिया")
        assert script == Script.DEVANAGARI
        assert proportion > 0.9

    def test_thai(self):
        """Test Thai script detection."""
        script, proportion = detect_script("สวัสดีโลก")
        assert script == Script.THAI
        assert proportion > 0.9

    def test_hangul(self):
        """Test Hangul (Korean) script detection."""
        script, proportion = detect_script("안녕하세요")
        assert script == Script.HANGUL
        assert proportion > 0.9

    def test_han(self):
        """Test Han (Chinese) script detection."""
        script, proportion = detect_script("你好世界")
        assert script == Script.HAN
        assert proportion > 0.9

    def test_hiragana(self):
        """Test Hiragana script detection."""
        script, proportion = detect_script("こんにちは")
        assert script == Script.HIRAGANA
        assert proportion > 0.9

    def test_mixed_scripts(self):
        """Test mixed script detection."""
        script, proportion = detect_script("Hello Привет مرحبا")
        assert script == Script.MIXED or proportion < 0.7

    def test_empty_string(self):
        """Test empty string."""
        script, proportion = detect_script("")
        assert script == Script.UNKNOWN
        assert proportion == 0.0


class TestScriptFilter:
    """Tests for ScriptFilter class."""

    def test_filter_cyrillic_languages(self):
        """Test filtering to Cyrillic languages."""
        filter = ScriptFilter()
        langs = filter.filter_languages("Привет мир")

        assert langs is not None
        assert "ru" in langs
        assert "uk" in langs
        assert "en" not in langs

    def test_filter_arabic_languages(self):
        """Test filtering to Arabic script languages."""
        filter = ScriptFilter()
        langs = filter.filter_languages("مرحبا")

        assert langs is not None
        assert "ar" in langs
        assert "fa" in langs

    def test_no_filter_latin(self):
        """Test that Latin script doesn't filter."""
        filter = ScriptFilter()
        langs = filter.filter_languages("Hello world")

        # Latin has too many languages, so no filtering
        assert langs is None

    def test_get_script_hint(self):
        """Test getting script hints."""
        filter = ScriptFilter()
        hints = filter.get_script_hint("Привет")

        assert hints is not None
        assert "ru" in hints
        assert hints["ru"] > 0

    def test_is_japanese(self):
        """Test Japanese detection."""
        filter = ScriptFilter()

        assert filter.is_japanese("こんにちは")
        assert filter.is_japanese("日本語テスト")
        assert not filter.is_japanese("Hello")

    def test_is_korean(self):
        """Test Korean detection."""
        filter = ScriptFilter()

        assert filter.is_korean("안녕하세요")
        assert not filter.is_korean("Hello")

    def test_is_chinese(self):
        """Test Chinese detection."""
        filter = ScriptFilter()

        assert filter.is_chinese("你好世界")
        # Japanese text with kana should not be Chinese
        assert not filter.is_chinese("こんにちは")
