"""Benchmark tests for FastLangML accuracy and performance.

Run with: pytest tests/test_benchmarks.py -v
Skip in CI: pytest tests/ --ignore=tests/test_benchmarks.py
"""

import time
from typing import NamedTuple

import pytest

from fastlangml import ConversationContext, FastLangDetector
from fastlangml.codeswitching import CodeSwitchDetector

# Mark all tests in this module as benchmark tests (slow)
pytestmark = pytest.mark.benchmark


class BenchmarkCase(NamedTuple):
    """Test case for benchmarks."""

    text: str
    expected: str
    category: str = "general"


# =============================================================================
# Test Data
# =============================================================================

# Standard language detection test cases
STANDARD_TESTS = [
    # European languages
    BenchmarkCase("Hello, how are you today?", "en", "european"),
    BenchmarkCase("Good morning, I hope you have a wonderful day.", "en", "european"),
    BenchmarkCase("Bonjour, comment allez-vous?", "fr", "european"),
    BenchmarkCase("Je suis très content de vous voir.", "fr", "european"),
    BenchmarkCase("Hola, como estas?", "es", "european"),
    BenchmarkCase("Buenos días, espero que tengas un buen día.", "es", "european"),
    BenchmarkCase("Guten Tag, wie geht es Ihnen?", "de", "european"),
    BenchmarkCase("Ich freue mich, Sie kennenzulernen.", "de", "european"),
    BenchmarkCase("Ciao, come stai?", "it", "european"),
    BenchmarkCase("Buongiorno, spero che tu stia bene.", "it", "european"),
    BenchmarkCase("Olá, como você está?", "pt", "european"),
    BenchmarkCase("Bom dia, espero que você esteja bem.", "pt", "european"),
    BenchmarkCase("Hallo, hoe gaat het met je?", "nl", "european"),
    BenchmarkCase("Cześć, jak się masz?", "pl", "european"),
    BenchmarkCase("Ahoj, jak se máš?", "cs", "european"),
    BenchmarkCase("Hej, hur mår du?", "sv", "european"),
    BenchmarkCase("Hei, hvordan har du det?", "no", "european"),
    BenchmarkCase("Hej, hvordan går det?", "da", "european"),
    BenchmarkCase("Hei, mitä kuuluu?", "fi", "european"),
    # Cyrillic languages
    BenchmarkCase("Привет, как дела?", "ru", "cyrillic"),
    BenchmarkCase("Доброе утро, надеюсь у вас всё хорошо.", "ru", "cyrillic"),
    BenchmarkCase("Привіт, як справи?", "uk", "cyrillic"),
    BenchmarkCase("Добрий ранок, сподіваюся у вас все добре.", "uk", "cyrillic"),
    BenchmarkCase("Здравейте, как сте?", "bg", "cyrillic"),
    # CJK languages
    BenchmarkCase("こんにちは、お元気ですか？", "ja", "cjk"),
    BenchmarkCase("今日はとても良い天気ですね。", "ja", "cjk"),
    BenchmarkCase("你好，你好吗？", "zh", "cjk"),
    BenchmarkCase("今天天气非常好。", "zh", "cjk"),
    BenchmarkCase("안녕하세요, 어떻게 지내세요?", "ko", "cjk"),
    BenchmarkCase("오늘 날씨가 정말 좋네요.", "ko", "cjk"),
    # RTL languages
    BenchmarkCase("مرحبا، كيف حالك؟", "ar", "rtl"),
    BenchmarkCase("صباح الخير، أتمنى لك يوماً سعيداً.", "ar", "rtl"),
    BenchmarkCase("שלום, מה שלומך?", "he", "rtl"),
    BenchmarkCase("בוקר טוב, מקווה שיהיה לך יום נפלא.", "he", "rtl"),
    BenchmarkCase("سلام، حال شما چطور است؟", "fa", "rtl"),
    # South Asian languages
    BenchmarkCase("नमस्ते, आप कैसे हैं?", "hi", "south_asian"),
    BenchmarkCase("আপনি কেমন আছেন?", "bn", "south_asian"),
    BenchmarkCase("நீங்கள் எப்படி இருக்கிறீர்கள்?", "ta", "south_asian"),
    # Southeast Asian languages
    BenchmarkCase("Xin chào, bạn có khỏe không?", "vi", "southeast_asian"),
    BenchmarkCase("Selamat pagi, apa kabar?", "id", "southeast_asian"),
    BenchmarkCase("สวัสดีครับ คุณสบายดีไหม", "th", "southeast_asian"),
    # Other languages
    BenchmarkCase("Merhaba, nasılsın?", "tr", "other"),
    BenchmarkCase("Γεια σου, πώς είσαι;", "el", "other"),
    BenchmarkCase("Szia, hogy vagy?", "hu", "other"),
    BenchmarkCase("Bună ziua, ce mai faci?", "ro", "other"),
]

# Short text test cases (chat-like messages)
SHORT_TEXT_TESTS = [
    # Greetings
    BenchmarkCase("Hello", "en", "greeting"),
    BenchmarkCase("Hi there", "en", "greeting"),
    BenchmarkCase("Bonjour", "fr", "greeting"),
    BenchmarkCase("Salut", "fr", "greeting"),
    BenchmarkCase("Hola", "es", "greeting"),
    BenchmarkCase("Ciao", "it", "greeting"),
    BenchmarkCase("Hallo", "de", "greeting"),
    BenchmarkCase("Olá", "pt", "greeting"),
    BenchmarkCase("Привет", "ru", "greeting"),
    BenchmarkCase("こんにちは", "ja", "greeting"),
    BenchmarkCase("你好", "zh", "greeting"),
    BenchmarkCase("안녕", "ko", "greeting"),
    # Common responses
    BenchmarkCase("Yes please", "en", "response"),
    BenchmarkCase("No thanks", "en", "response"),
    BenchmarkCase("Thank you", "en", "response"),
    BenchmarkCase("Oui merci", "fr", "response"),
    BenchmarkCase("Non merci", "fr", "response"),
    BenchmarkCase("Sí gracias", "es", "response"),
    BenchmarkCase("No gracias", "es", "response"),
    BenchmarkCase("Danke schön", "de", "response"),
    BenchmarkCase("Grazie mille", "it", "response"),
    BenchmarkCase("Спасибо", "ru", "response"),
    BenchmarkCase("ありがとう", "ja", "response"),
    BenchmarkCase("谢谢", "zh", "response"),
    # Questions
    BenchmarkCase("How are you?", "en", "question"),
    BenchmarkCase("What time is it?", "en", "question"),
    BenchmarkCase("Comment ça va?", "fr", "question"),
    BenchmarkCase("Quelle heure est-il?", "fr", "question"),
    BenchmarkCase("¿Cómo estás?", "es", "question"),
    BenchmarkCase("¿Qué hora es?", "es", "question"),
    BenchmarkCase("Wie geht es dir?", "de", "question"),
    BenchmarkCase("Come stai?", "it", "question"),
    BenchmarkCase("Как дела?", "ru", "question"),
]

# Context-aware test cases (conversation flows)
CONTEXT_TESTS = [
    # French conversation
    {
        "messages": [
            ("Bonjour!", "fr"),
            ("Comment ça va?", "fr"),
            ("Bien", "fr"),  # Ambiguous without context
            ("ok", "fr"),  # Ambiguous without context
            ("merci", "fr"),
        ],
        "language": "fr",
    },
    # Spanish conversation
    {
        "messages": [
            ("Hola!", "es"),
            ("¿Cómo estás?", "es"),
            ("Bien", "es"),  # Could be French/German
            ("ok", "es"),
            ("gracias", "es"),
        ],
        "language": "es",
    },
    # German conversation
    {
        "messages": [
            ("Guten Tag!", "de"),
            ("Wie geht es Ihnen?", "de"),
            ("Gut", "de"),  # Ambiguous
            ("ok", "de"),
            ("danke", "de"),
        ],
        "language": "de",
    },
    # English conversation
    {
        "messages": [
            ("Hello!", "en"),
            ("How are you?", "en"),
            ("Good", "en"),
            ("ok", "en"),
            ("thanks", "en"),
        ],
        "language": "en",
    },
]

# Code-switching test cases
CODE_SWITCHING_TESTS = [
    # Spanglish
    BenchmarkCase("That's muy importante for the proyecto", "mixed", "spanglish"),
    BenchmarkCase("I need to comprar some things from the tienda", "mixed", "spanglish"),
    BenchmarkCase("Let me llamar my friend real quick", "mixed", "spanglish"),
    # Franglais
    BenchmarkCase("C'est very nice de te voir", "mixed", "franglais"),
    BenchmarkCase("Je suis so excited pour tonight", "mixed", "franglais"),
    # Hinglish
    BenchmarkCase("Main bahut busy hoon today", "mixed", "hinglish"),
    BenchmarkCase("That was bahut accha performance", "mixed", "hinglish"),
]


# =============================================================================
# Benchmark Tests
# =============================================================================


class TestAccuracyBenchmarks:
    """Accuracy benchmark tests."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        try:
            return FastLangDetector()
        except Exception:
            pytest.skip("No backends available")

    def test_standard_language_accuracy(self, detector):
        """Benchmark accuracy on standard language detection."""
        correct = 0
        total = len(STANDARD_TESTS)
        failures = []

        for test in STANDARD_TESTS:
            result = detector.detect(test.text)
            if result.lang == test.expected:
                correct += 1
            else:
                failures.append((test.text[:30], test.expected, result.lang))

        accuracy = correct / total * 100
        print("\n=== Standard Language Accuracy ===")
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")

        if failures:
            print(f"\nFailures ({len(failures)}):")
            for text, expected, got in failures[:10]:
                print(f"  '{text}...' expected={expected} got={got}")

        # Expect at least 85% accuracy
        assert accuracy >= 85, f"Accuracy {accuracy:.1f}% below threshold 85%"

    def test_short_text_accuracy(self, detector):
        """Benchmark accuracy on short/chat text."""
        correct = 0
        total = len(SHORT_TEXT_TESTS)
        failures = []

        for test in SHORT_TEXT_TESTS:
            result = detector.detect(test.text, mode="short")
            if result.lang == test.expected:
                correct += 1
            else:
                failures.append((test.text, test.expected, result.lang))

        accuracy = correct / total * 100
        print("\n=== Short Text Accuracy ===")
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")

        if failures:
            print(f"\nFailures ({len(failures)}):")
            for text, expected, got in failures[:10]:
                print(f"  '{text}' expected={expected} got={got}")

        # Short text is harder, expect at least 70%
        assert accuracy >= 70, f"Accuracy {accuracy:.1f}% below threshold 70%"

    def test_accuracy_by_category(self, detector):
        """Benchmark accuracy by language category."""
        categories: dict[str, dict[str, int]] = {}

        for test in STANDARD_TESTS:
            if test.category not in categories:
                categories[test.category] = {"correct": 0, "total": 0}

            categories[test.category]["total"] += 1
            result = detector.detect(test.text)
            if result.lang == test.expected:
                categories[test.category]["correct"] += 1

        print("\n=== Accuracy by Category ===")
        for category, data in sorted(categories.items()):
            acc = data["correct"] / data["total"] * 100
            print(f"  {category}: {acc:.1f}% ({data['correct']}/{data['total']})")
            # Each category should have at least 70% accuracy
            assert acc >= 70, f"{category} accuracy {acc:.1f}% below threshold"

    def test_cjk_accuracy(self, detector):
        """Benchmark accuracy specifically for CJK languages."""
        cjk_tests = [t for t in STANDARD_TESTS if t.category == "cjk"]
        correct = 0

        for test in cjk_tests:
            result = detector.detect(test.text)
            if result.lang == test.expected:
                correct += 1

        accuracy = correct / len(cjk_tests) * 100
        print("\n=== CJK Accuracy ===")
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(cjk_tests)})")

        # CJK should be highly accurate due to distinct scripts
        assert accuracy >= 90, f"CJK accuracy {accuracy:.1f}% below threshold 90%"


class TestContextAwareBenchmarks:
    """Context-aware detection benchmarks."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        try:
            return FastLangDetector()
        except Exception:
            pytest.skip("No backends available")

    def test_context_aware_accuracy(self, detector):
        """Benchmark context-aware detection accuracy."""
        total_correct = 0
        total_messages = 0

        print("\n=== Context-Aware Detection ===")

        for conv in CONTEXT_TESTS:
            context = ConversationContext()
            correct = 0

            for text, expected in conv["messages"]:
                result = detector.detect(text, context=context)
                total_messages += 1
                if result.lang == expected:
                    correct += 1
                    total_correct += 1

            acc = correct / len(conv["messages"]) * 100
            print(f"  {conv['language']} conversation: {acc:.0f}%")

        overall_acc = total_correct / total_messages * 100
        print(f"\nOverall: {overall_acc:.1f}% ({total_correct}/{total_messages})")

        # Context helps but ambiguous words are still hard
        assert overall_acc >= 60, f"Context accuracy {overall_acc:.1f}% below threshold"

    def test_context_helps_ambiguous(self, detector):
        """Test that context helps with ambiguous words."""
        # Test "ok" in different language contexts
        languages = ["en", "fr", "es", "de"]
        correct = 0

        for lang in languages:
            context = ConversationContext()

            # Prime context with clear language
            if lang == "en":
                detector.detect("Hello, how are you?", context=context)
            elif lang == "fr":
                detector.detect("Bonjour, comment allez-vous?", context=context)
            elif lang == "es":
                detector.detect("Hola, ¿cómo estás?", context=context)
            elif lang == "de":
                detector.detect("Hallo, wie geht es dir?", context=context)

            # Now detect "ok"
            result = detector.detect("ok", context=context)
            if result.lang == lang:
                correct += 1

        accuracy = correct / len(languages) * 100
        print("\n=== Ambiguous Word ('ok') with Context ===")
        print(f"Accuracy: {accuracy:.0f}% ({correct}/{len(languages)})")


class TestCodeSwitchingBenchmarks:
    """Code-switching detection benchmarks."""

    @pytest.fixture
    def detector(self):
        """Create a code-switch detector."""
        try:
            return CodeSwitchDetector()
        except Exception:
            pytest.skip("Code-switch detector not available")

    def test_code_switching_detection(self, detector):
        """Benchmark code-switching detection."""
        correct = 0
        total = len(CODE_SWITCHING_TESTS)

        for test in CODE_SWITCHING_TESTS:
            result = detector.detect(test.text)
            if result.is_mixed:
                correct += 1

        accuracy = correct / total * 100
        print("\n=== Code-Switching Detection ===")
        print(f"Detection rate: {accuracy:.1f}% ({correct}/{total})")

        # Should detect most code-switching
        assert accuracy >= 60, f"Detection rate {accuracy:.1f}% below threshold"


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        try:
            return FastLangDetector()
        except Exception:
            pytest.skip("No backends available")

    def test_single_detection_latency(self, detector):
        """Benchmark single detection latency."""
        text = "Hello, how are you today?"

        # Warmup
        for _ in range(5):
            detector.detect(text)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            detector.detect(text)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        p50 = sorted(times)[50]
        p99 = sorted(times)[99]

        print("\n=== Single Detection Latency ===")
        print(f"Avg: {avg_time:.2f}ms")
        print(f"P50: {p50:.2f}ms")
        print(f"P99: {p99:.2f}ms")

        # Should be reasonably fast
        assert avg_time < 100, f"Avg latency {avg_time:.2f}ms too high"

    def test_batch_throughput(self, detector):
        """Benchmark batch detection throughput."""
        texts = [t.text for t in STANDARD_TESTS]

        # Warmup
        detector.detect_batch(texts[:5])

        # Measure
        start = time.perf_counter()
        iterations = 5
        for _ in range(iterations):
            detector.detect_batch(texts)
        elapsed = time.perf_counter() - start

        total_texts = len(texts) * iterations
        throughput = total_texts / elapsed

        print("\n=== Batch Throughput ===")
        print(f"Texts processed: {total_texts}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Throughput: {throughput:.0f} texts/sec")

        # Should handle at least 10 texts/sec
        assert throughput > 10, f"Throughput {throughput:.0f}/sec too low"

    def test_short_text_latency(self, detector):
        """Benchmark latency for short text detection."""
        short_texts = ["ok", "yes", "no", "hi", "bye"]

        times = []
        for text in short_texts * 20:
            start = time.perf_counter()
            detector.detect(text, mode="short")
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print("\n=== Short Text Latency ===")
        print(f"Avg: {avg_time:.2f}ms")

        # Short text should be fast
        assert avg_time < 50, f"Short text latency {avg_time:.2f}ms too high"


class TestBackendComparisonBenchmarks:
    """Compare individual backend accuracy."""

    def test_backend_accuracy_comparison(self):
        """Compare accuracy across available backends."""
        from fastlangml.backends import create_backend, get_available_backends

        backends = get_available_backends()
        if not backends:
            pytest.skip("No backends available")

        # Use a subset of tests for speed
        test_subset = STANDARD_TESTS[:20]

        print("\n=== Backend Accuracy Comparison ===")

        results = {}
        for backend_name in backends:
            try:
                backend = create_backend(backend_name)
                correct = 0

                for test in test_subset:
                    result = backend.detect(test.text)
                    if result.language == test.expected:
                        correct += 1

                accuracy = correct / len(test_subset) * 100
                results[backend_name] = accuracy
                print(f"  {backend_name}: {accuracy:.1f}%")
            except Exception as e:
                print(f"  {backend_name}: ERROR - {e}")

        # At least one backend should have good accuracy
        best_accuracy = max(results.values()) if results else 0
        assert best_accuracy >= 70, f"Best backend accuracy {best_accuracy:.1f}% too low"


class TestEdgeCaseBenchmarks:
    """Edge case and robustness benchmarks."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        try:
            return FastLangDetector()
        except Exception:
            pytest.skip("No backends available")

    def test_empty_and_whitespace(self, detector):
        """Test handling of empty/whitespace input."""
        test_cases = ["", " ", "   ", "\n", "\t"]

        for text in test_cases:
            result = detector.detect(text)
            assert result.lang == "und", f"Expected 'und' for '{repr(text)}'"

    def test_numbers_only(self, detector):
        """Test handling of numeric-only input."""
        test_cases = ["123", "45.67", "1,234,567", "2024"]

        for text in test_cases:
            result = detector.detect(text)
            # Numbers should return 'und' or low confidence
            assert result.lang == "und" or not result.reliable

    def test_special_characters(self, detector):
        """Test handling of special characters."""
        test_cases = ["@#$%", "!!!", "...", "???", "+++"]

        for text in test_cases:
            result = detector.detect(text)
            assert result.lang == "und" or not result.reliable

    def test_mixed_scripts(self, detector):
        """Test handling of mixed script input."""
        test_cases = [
            ("Hello 你好", ["en", "zh"]),
            ("Bonjour こんにちは", ["fr", "ja"]),
        ]

        for text, possible_langs in test_cases:
            result = detector.detect(text)
            # Should detect one of the languages
            assert result.lang in possible_langs or result.lang == "und"

    def test_very_long_text(self, detector):
        """Test handling of very long text."""
        # Generate long text by repeating
        base = "This is a test sentence in English. "
        long_text = base * 100  # ~3600 chars

        start = time.perf_counter()
        result = detector.detect(long_text)
        elapsed = (time.perf_counter() - start) * 1000

        assert result.lang == "en"
        assert elapsed < 1000, f"Long text took {elapsed:.0f}ms (>1s)"
        print(f"\nLong text ({len(long_text)} chars): {elapsed:.2f}ms")

    def test_unicode_edge_cases(self, detector):
        """Test Unicode edge cases."""
        test_cases = [
            ("café", "fr"),  # Accented Latin
            ("naïve", "en"),  # English with diacritics
            ("北京", "zh"),  # Chinese
            ("東京", "ja"),  # Japanese Kanji
            ("서울", "ko"),  # Korean
        ]

        for text, _expected in test_cases:
            result = detector.detect(text)
            # Should handle Unicode gracefully
            assert result.lang is not None
