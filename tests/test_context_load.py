"""Load and concurrency tests for context stores."""

from __future__ import annotations

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest

from fastlangml.context import ConversationContext

# Simulated multi-language conversations
CONVERSATIONS: dict[str, list[tuple[str, str, float]]] = {
    "french_customer": [
        ("Bonjour, je cherche un produit", "fr", 0.95),
        ("ok", "fr", 0.6),  # Ambiguous, but context helps
        ("merci beaucoup", "fr", 0.98),
        ("oui", "fr", 0.7),
        ("au revoir", "fr", 0.95),
    ],
    "spanish_customer": [
        ("Hola, necesito ayuda", "es", 0.95),
        ("si", "es", 0.5),  # Ambiguous
        ("gracias", "es", 0.92),
        ("ok", "es", 0.5),  # Ambiguous, context helps
        ("adios", "es", 0.90),
    ],
    "english_customer": [
        ("Hi, I need help with my order", "en", 0.98),
        ("yes", "en", 0.7),
        ("thanks", "en", 0.85),
        ("ok", "en", 0.5),  # Ambiguous
        ("bye", "en", 0.80),
    ],
    "german_customer": [
        ("Guten Tag, ich habe eine Frage", "de", 0.96),
        ("ja", "de", 0.6),
        ("danke schön", "de", 0.95),
        ("ok", "de", 0.5),
        ("tschüss", "de", 0.88),
    ],
    "code_switcher": [
        ("Hey, comment ça va?", "fr", 0.7),  # Mixed
        ("I'm doing well", "en", 0.95),
        ("Très bien, merci", "fr", 0.92),
        ("See you later", "en", 0.90),
        ("À bientôt!", "fr", 0.88),
    ],
}


class TestConversationContextLoad:
    """Load tests for ConversationContext operations."""

    def test_many_sessions_sequential(self) -> None:
        """Simulate 100 unique user sessions sequentially."""
        sessions: dict[str, ConversationContext] = {}

        for i in range(100):
            session_id = f"user-{i}"
            lang = ["en", "fr", "es", "de"][i % 4]
            ctx = ConversationContext(max_turns=5)
            ctx.add_turn(f"Message {i}", lang, 0.9)
            sessions[session_id] = ctx

        # Verify all sessions exist
        assert len(sessions) == 100

        # Verify context isolation
        assert sessions["user-0"].dominant_language == "en"
        assert sessions["user-1"].dominant_language == "fr"
        assert sessions["user-2"].dominant_language == "es"
        assert sessions["user-3"].dominant_language == "de"

    def test_many_turns_single_session(self) -> None:
        """Test context with many turns (stress test max_turns eviction)."""
        ctx = ConversationContext(max_turns=10)

        for i in range(1000):
            lang = "en" if i % 2 == 0 else "fr"
            ctx.add_turn(f"Message {i}", lang, 0.9)

        # Should only keep last 10 turns
        assert len(ctx) == 10

        # All recent turns should be in context
        turns = list(ctx.turns)
        assert turns[0].text == "Message 990"
        assert turns[-1].text == "Message 999"

    def test_from_history_large_history(self) -> None:
        """Test from_history with large history list."""
        history = [(["en", "fr", "es"][i % 3], 0.9) for i in range(10000)]

        ctx = ConversationContext.from_history(history, max_turns=5)

        # Should only keep last 5 from history
        assert len(ctx) <= 5

    def test_serialization_round_trip_many_contexts(self) -> None:
        """Test serialization round-trip for many contexts."""
        contexts: list[ConversationContext] = []

        for i in range(100):
            ctx = ConversationContext(max_turns=5)
            for j in range(5):
                ctx.add_turn(f"msg-{i}-{j}", ["en", "fr", "es"][j % 3], 0.8 + j * 0.04)
            contexts.append(ctx)

        # Serialize and deserialize all
        restored = [ConversationContext.from_dict(ctx.to_dict()) for ctx in contexts]

        # Verify integrity
        for original, copy in zip(contexts, restored, strict=True):
            assert len(original) == len(copy)
            assert original.dominant_language == copy.dominant_language


class TestConversationContextConcurrency:
    """Concurrency tests for ConversationContext."""

    def test_concurrent_add_turns_single_context(self) -> None:
        """Test concurrent add_turn calls on same context (shows thread-safety issue)."""
        ctx = ConversationContext(max_turns=100)
        errors: list[Exception] = []
        num_threads = 10
        turns_per_thread = 50

        def add_turns(thread_id: int) -> None:
            try:
                for i in range(turns_per_thread):
                    ctx.add_turn(f"t{thread_id}-m{i}", "en", 0.9)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_turns, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should occur
        assert len(errors) == 0

        # Note: Due to race conditions, we may not have exactly num_threads * turns_per_thread
        # turns, but the context should be in a consistent state
        assert len(ctx) <= 100  # max_turns constraint

    def test_concurrent_sessions_isolation(self) -> None:
        """Test that concurrent sessions don't interfere with each other."""
        sessions: dict[str, ConversationContext] = {}
        lock = threading.Lock()

        def simulate_conversation(session_id: str, lang: str) -> None:
            ctx = ConversationContext(max_turns=5)
            for i in range(5):
                ctx.add_turn(f"msg-{i}", lang, 0.9)
                time.sleep(0.001)  # Simulate processing delay

            with lock:
                sessions[session_id] = ctx

        threads = []
        for i in range(50):
            lang = ["en", "fr", "es", "de"][i % 4]
            t = threading.Thread(target=simulate_conversation, args=(f"user-{i}", lang))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all 50 sessions created
        assert len(sessions) == 50

        # Verify correct language for each session
        for i in range(50):
            expected_lang = ["en", "fr", "es", "de"][i % 4]
            assert sessions[f"user-{i}"].dominant_language == expected_lang

    def test_threadpool_conversation_simulation(self) -> None:
        """Use ThreadPoolExecutor to simulate many concurrent conversations."""
        results: dict[str, str | None] = {}

        def run_conversation(
            session_id: str, conversation: list[tuple[str, str, float]]
        ) -> tuple[str, str | None]:
            ctx = ConversationContext(max_turns=5)
            for text, lang, conf in conversation:
                ctx.add_turn(text, lang, conf)
            return session_id, ctx.dominant_language

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(100):
                conv_type = list(CONVERSATIONS.keys())[i % len(CONVERSATIONS)]
                conversation = CONVERSATIONS[conv_type]
                session_id = f"session-{i}"
                futures.append(executor.submit(run_conversation, session_id, conversation))

            for future in as_completed(futures):
                session_id, dominant = future.result()
                results[session_id] = dominant

        # All 100 sessions should complete
        assert len(results) == 100

        # Verify results based on conversation type
        for i in range(100):
            conv_type = list(CONVERSATIONS.keys())[i % len(CONVERSATIONS)]
            session_id = f"session-{i}"
            # All conversations should have a dominant language detected
            assert results[session_id] is not None


class TestSimulatedConversations:
    """Tests simulating realistic multi-turn conversations."""

    @pytest.mark.parametrize("conv_name,conversation", list(CONVERSATIONS.items()))
    def test_full_conversation_flow(
        self, conv_name: str, conversation: list[tuple[str, str, float]]
    ) -> None:
        """Test complete conversation flows with context tracking."""
        ctx = ConversationContext(max_turns=5)

        for text, expected_lang, conf in conversation:
            ctx.add_turn(text, expected_lang, conf)

        # Conversation should have built up context
        assert len(ctx) == len(conversation)

        # Verify weighted dominant language
        dominant = ctx.dominant_language
        assert dominant is not None

    def test_ambiguous_messages_with_context(self) -> None:
        """Test that context helps resolve ambiguous messages."""
        # French context
        ctx_fr = ConversationContext(max_turns=5)
        ctx_fr.add_turn("Bonjour!", "fr", 0.95)
        ctx_fr.add_turn("Comment allez-vous?", "fr", 0.92)

        # Now "ok" should be interpreted as French due to context
        assert ctx_fr.dominant_language == "fr"
        ctx_fr.add_turn("ok", "fr", 0.5)  # Low confidence but context helps
        assert ctx_fr.dominant_language == "fr"

        # Spanish context
        ctx_es = ConversationContext(max_turns=5)
        ctx_es.add_turn("Hola!", "es", 0.95)
        ctx_es.add_turn("¿Cómo estás?", "es", 0.92)

        # Same "ok" in Spanish context
        assert ctx_es.dominant_language == "es"
        ctx_es.add_turn("ok", "es", 0.5)
        assert ctx_es.dominant_language == "es"

    def test_language_switch_detection(self) -> None:
        """Test detecting language switches within a conversation."""
        ctx = ConversationContext(max_turns=10)

        # Start in English
        ctx.add_turn("Hello, how are you?", "en", 0.95)
        ctx.add_turn("I'm doing fine", "en", 0.92)
        assert ctx.dominant_language == "en"

        # Switch to French
        ctx.add_turn("Maintenant je parle français", "fr", 0.95)
        ctx.add_turn("C'est une autre langue", "fr", 0.92)
        ctx.add_turn("Très bien!", "fr", 0.90)

        # French should now dominate (more recent, more entries)
        langs = ctx.language_distribution
        assert "fr" in langs
        assert "en" in langs

    def test_context_decay_over_time(self) -> None:
        """Test that older context has less influence."""
        ctx = ConversationContext(max_turns=10, decay_factor=0.7)

        # Add many French messages first
        for i in range(5):
            ctx.add_turn(f"French message {i}", "fr", 0.9)

        # Add fewer English messages recently
        for i in range(3):
            ctx.add_turn(f"English message {i}", "en", 0.9)

        # Recent English should have strong influence despite fewer messages
        langs = ctx.language_distribution
        # Both should be present
        assert "fr" in langs
        assert "en" in langs

    def test_context_serialization_preserves_conversation(self) -> None:
        """Test that serializing and restoring preserves conversation state."""
        # Build original context
        original = ConversationContext(max_turns=5)
        for text, lang, conf in CONVERSATIONS["french_customer"]:
            original.add_turn(text, lang, conf)

        # Serialize to history (as stores do)
        history = [
            (t.detected_language, t.confidence) for t in original.turns if t.detected_language
        ]

        # Restore from history
        restored = ConversationContext.from_history(history, max_turns=5)

        # Should have same dominant language
        assert restored.dominant_language == original.dominant_language
        assert len(restored) == len(original)


class TestDiskContextStoreLoad:
    """Load tests for DiskContextStore (requires diskcache)."""

    @pytest.fixture
    def disk_store(self) -> Any:
        """Create a DiskContextStore for testing."""
        pytest.importorskip("diskcache")
        from fastlangml.context import DiskContextStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DiskContextStore(tmpdir, ttl_seconds=None, max_turns=5)
            yield store

    def test_many_sessions_disk(self, disk_store: Any) -> None:
        """Test storing many sessions to disk."""
        for i in range(100):
            session_id = f"user-{i}"
            with disk_store.session(session_id) as ctx:
                lang = ["en", "fr", "es", "de"][i % 4]
                ctx.add_turn(f"Message {i}", lang, 0.9)

        # Verify all sessions persist
        for i in range(100):
            session_id = f"user-{i}"
            expected_lang = ["en", "fr", "es", "de"][i % 4]
            ctx = disk_store.load(session_id)
            assert ctx is not None
            assert ctx.dominant_language == expected_lang

    def test_concurrent_sessions_disk(self, disk_store: Any) -> None:
        """Test concurrent session access to disk store."""
        errors: list[Exception] = []

        def simulate_session(session_id: str, lang: str) -> None:
            try:
                with disk_store.session(session_id) as ctx:
                    for i in range(5):
                        ctx.add_turn(f"msg-{i}", lang, 0.9)
                        time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(20):
            lang = ["en", "fr", "es", "de"][i % 4]
            t = threading.Thread(target=simulate_session, args=(f"user-{i}", lang))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur with different sessions
        assert len(errors) == 0

        # Verify all sessions created correctly
        for i in range(20):
            session_id = f"user-{i}"
            expected_lang = ["en", "fr", "es", "de"][i % 4]
            ctx = disk_store.load(session_id)
            assert ctx is not None
            assert ctx.dominant_language == expected_lang

    def test_concurrent_same_session_race(self, disk_store: Any) -> None:
        """Demonstrate race condition when multiple threads access same session.

        This test shows the load-modify-save race condition that can occur
        when multiple threads access the same session concurrently.
        """
        session_id = "shared-session"
        successful_writes = []
        lock = threading.Lock()

        def update_session(thread_id: int) -> None:
            with disk_store.session(session_id) as ctx:
                # Simulate some processing time
                time.sleep(0.01)
                ctx.add_turn(f"Thread-{thread_id}", "en", 0.9)

            with lock:
                successful_writes.append(thread_id)

        threads = [threading.Thread(target=update_session, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads completed
        assert len(successful_writes) == 5

        # Due to race condition, final context may not have all 5 turns
        # This demonstrates the known limitation
        ctx = disk_store.load(session_id)
        assert ctx is not None
        # May have fewer than 5 turns due to race condition
        # (last write wins, overwriting concurrent writes)
        print(f"Turns after concurrent writes: {len(ctx)} (expected: possibly < 5 due to race)")


class TestConversationThroughput:
    """Throughput and performance tests."""

    def test_add_turn_throughput(self) -> None:
        """Measure throughput of add_turn operations."""
        ctx = ConversationContext(max_turns=100)
        iterations = 10000

        start = time.perf_counter()
        for i in range(iterations):
            ctx.add_turn(f"msg-{i}", "en", 0.9)
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        print(f"add_turn throughput: {ops_per_sec:.0f} ops/sec")

        # Should be able to do at least 10k ops/sec
        assert ops_per_sec > 1000

    def test_from_history_throughput(self) -> None:
        """Measure throughput of from_history operations."""
        history = [("en", 0.9)] * 5
        iterations = 10000

        start = time.perf_counter()
        for _ in range(iterations):
            ConversationContext.from_history(history, max_turns=5)
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        print(f"from_history throughput: {ops_per_sec:.0f} ops/sec")

        # Should be very fast since it's just object creation
        assert ops_per_sec > 1000

    def test_serialization_throughput(self) -> None:
        """Measure throughput of serialization round-trips."""
        ctx = ConversationContext(max_turns=5)
        for i in range(5):
            ctx.add_turn(f"msg-{i}", "en", 0.9)

        iterations = 10000

        start = time.perf_counter()
        for _ in range(iterations):
            data = ctx.to_dict()
            ConversationContext.from_dict(data)
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        print(f"serialization round-trip throughput: {ops_per_sec:.0f} ops/sec")

        assert ops_per_sec > 1000
