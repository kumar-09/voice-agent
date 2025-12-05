"""Tests for the intelligent interruption filter.

This module tests the InterruptionFilter class which distinguishes between
passive acknowledgements and active interruptions based on whether the agent
is speaking.
"""

import sys
import types

import pytest


# Helper to create a minimal filter implementation for testing without importing the full module
def create_test_filter():
    """Create a test filter instance by executing the module code directly."""
    import os

    # Create a mock module for the namespace to satisfy dataclass decorator
    mock_module = types.ModuleType("interruption_filter")
    mock_module.__dict__["__builtins__"] = __builtins__
    sys.modules["test_interruption_filter_module"] = mock_module

    # Get the path to the module - use absolute path from workspace root
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_path = os.path.join(
        base_path,
        "livekit-agents",
        "livekit",
        "agents",
        "voice",
        "transcription",
        "interruption_filter.py",
    )

    with open(module_path, encoding="utf-8") as f:
        code = f.read()

    # Update __name__ to use our mock module
    mock_module.__dict__["__name__"] = "test_interruption_filter_module"
    exec(code, mock_module.__dict__)

    return (
        mock_module.__dict__["InterruptionFilter"],
        mock_module.__dict__["InterruptionFilterConfig"],
        mock_module.__dict__["DEFAULT_IGNORE_WORDS"],
        mock_module.__dict__["DEFAULT_INTERRUPT_KEYWORDS"],
    )


@pytest.fixture
def filter_classes():
    """Fixture to provide the filter classes."""
    return create_test_filter()


@pytest.fixture
def default_filter(filter_classes):
    """Fixture to provide a default configured filter."""
    InterruptionFilter, _, _, _ = filter_classes
    return InterruptionFilter()


class TestInterruptionFilterBasics:
    """Test basic functionality of the InterruptionFilter."""

    def test_filter_creation(self, filter_classes):
        """Test that filter can be created with default config."""
        InterruptionFilter, InterruptionFilterConfig, _, _ = filter_classes
        f = InterruptionFilter()
        assert f.config.enabled is True

    def test_filter_with_custom_config(self, filter_classes):
        """Test filter with custom configuration."""
        InterruptionFilter, InterruptionFilterConfig, _, _ = filter_classes
        config = InterruptionFilterConfig(
            ignore_words=frozenset(["custom1", "custom2"]),
            interrupt_keywords=frozenset(["override"]),
            enabled=True,
        )
        f = InterruptionFilter(config)
        assert f.config.ignore_words == frozenset(["custom1", "custom2"])
        assert f.config.interrupt_keywords == frozenset(["override"])

    def test_disabled_filter_always_allows_interrupt(self, filter_classes):
        """Test that disabled filter always returns True."""
        InterruptionFilter, InterruptionFilterConfig, _, _ = filter_classes
        config = InterruptionFilterConfig(enabled=False)
        f = InterruptionFilter(config)

        # Should always return True when disabled
        assert f.should_interrupt("yeah", agent_is_speaking=True) is True
        assert f.should_interrupt("yeah", agent_is_speaking=False) is True


class TestPassiveAcknowledgements:
    """Test that passive acknowledgements are correctly handled."""

    @pytest.mark.parametrize(
        "word",
        [
            "yeah",
            "yes",
            "yep",
            "yup",
            "ok",
            "okay",
            "hmm",
            "mhm",
            "uh-huh",
            "right",
            "aha",
            "i see",
            "sure",
            "got it",
            "alright",
            "cool",
            "nice",
            "great",
            "good",
            "fine",
        ],
    )
    def test_single_filler_word_ignored_when_speaking(self, default_filter, word):
        """Test that single filler words are ignored when agent is speaking."""
        result = default_filter.should_interrupt(word, agent_is_speaking=True)
        assert result is False, f"'{word}' should be ignored when agent is speaking"

    @pytest.mark.parametrize(
        "word",
        [
            "yeah",
            "yes",
            "ok",
            "hmm",
            "right",
        ],
    )
    def test_filler_words_processed_when_silent(self, default_filter, word):
        """Test that filler words trigger response when agent is silent."""
        result = default_filter.should_interrupt(word, agent_is_speaking=False)
        assert result is True, f"'{word}' should trigger response when agent is silent"

    def test_multiple_filler_words_ignored(self, default_filter):
        """Test that multiple filler words are ignored when agent is speaking."""
        result = default_filter.should_interrupt("yeah ok hmm", agent_is_speaking=True)
        assert result is False, "Multiple filler words should be ignored"

    def test_case_insensitive(self, default_filter):
        """Test that matching is case insensitive."""
        assert default_filter.should_interrupt("YEAH", agent_is_speaking=True) is False
        assert default_filter.should_interrupt("Yeah", agent_is_speaking=True) is False
        assert default_filter.should_interrupt("YeAh", agent_is_speaking=True) is False


class TestInterruptKeywords:
    """Test that interrupt keywords are correctly detected."""

    @pytest.mark.parametrize(
        "keyword",
        [
            "stop",
            "wait",
            "no",
            "nope",
            "pause",
            "actually",
            "but",
            "however",
            "excuse me",
            "sorry",
            "hold on",
        ],
    )
    def test_interrupt_keywords_work_when_speaking(self, default_filter, keyword):
        """Test that interrupt keywords trigger interruption when agent is speaking."""
        result = default_filter.should_interrupt(keyword, agent_is_speaking=True)
        assert result is True, f"'{keyword}' should trigger interruption"

    def test_interrupt_keywords_work_when_silent(self, default_filter):
        """Test that interrupt keywords trigger response when agent is silent."""
        result = default_filter.should_interrupt("stop", agent_is_speaking=False)
        assert result is True


class TestMixedInput:
    """Test handling of mixed input (filler + interrupt keywords)."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "yeah but wait",
            "ok but actually",
            "hmm wait a second",
            "yeah no stop",
            "ok hold on",
            "yeah actually",
        ],
    )
    def test_mixed_input_with_interrupt_keyword(self, default_filter, phrase):
        """Test that mixed input with interrupt keywords triggers interruption."""
        result = default_filter.should_interrupt(phrase, agent_is_speaking=True)
        assert result is True, f"'{phrase}' should trigger interruption due to keyword"

    @pytest.mark.parametrize(
        "phrase",
        [
            "yeah what",
            "ok tell me",
            "hmm explain",
            "right so",
        ],
    )
    def test_filler_with_other_words_interrupts(self, default_filter, phrase):
        """Test that filler words mixed with other words trigger interruption."""
        result = default_filter.should_interrupt(phrase, agent_is_speaking=True)
        assert result is True, f"'{phrase}' should trigger interruption"


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string(self, default_filter):
        """Test that empty string doesn't trigger interruption."""
        result = default_filter.should_interrupt("", agent_is_speaking=True)
        assert result is False

    def test_whitespace_only(self, default_filter):
        """Test that whitespace-only string doesn't trigger interruption."""
        result = default_filter.should_interrupt("   ", agent_is_speaking=True)
        assert result is False

    def test_phrase_in_ignore_list(self, default_filter):
        """Test that multi-word phrases in ignore list are recognized."""
        result = default_filter.should_interrupt("i see", agent_is_speaking=True)
        assert result is False

    def test_partial_phrase_match(self, default_filter):
        """Test that partial phrase matches work correctly."""
        # "hold on" is an interrupt keyword
        result = default_filter.should_interrupt("hold on", agent_is_speaking=True)
        assert result is True

    def test_update_config(self, filter_classes):
        """Test that config can be updated after creation."""
        InterruptionFilter, InterruptionFilterConfig, _, _ = filter_classes
        f = InterruptionFilter()

        new_config = InterruptionFilterConfig(
            ignore_words=frozenset(["custom"]), interrupt_keywords=frozenset(["override"])
        )
        f.update_config(new_config)

        assert f.config.ignore_words == frozenset(["custom"])


class TestScenarios:
    """Test complete scenarios from the issue description."""

    def test_scenario_1_long_explanation(self, default_filter):
        """Scenario 1: User says 'yeah/ok/hmm' while agent is speaking - should be ignored."""
        # Agent is reading a long paragraph
        assert default_filter.should_interrupt("okay", agent_is_speaking=True) is False
        assert default_filter.should_interrupt("yeah", agent_is_speaking=True) is False
        assert default_filter.should_interrupt("uh-huh", agent_is_speaking=True) is False

    def test_scenario_2_passive_affirmation(self, default_filter):
        """Scenario 2: Agent asks question, goes silent, user says 'Yeah' - should respond."""
        # Agent asks "Are you ready?" and goes silent
        result = default_filter.should_interrupt("yeah", agent_is_speaking=False)
        assert result is True, "Should process 'yeah' as valid answer when agent is silent"

    def test_scenario_3_correction(self, default_filter):
        """Scenario 3: User says 'No stop' while agent is counting - should interrupt."""
        result = default_filter.should_interrupt("no stop", agent_is_speaking=True)
        assert result is True, "'no stop' should interrupt immediately"

    def test_scenario_4_mixed_input(self, default_filter):
        """Scenario 4: User says 'Yeah okay but wait' - should interrupt due to 'but wait'."""
        result = default_filter.should_interrupt("yeah okay but wait", agent_is_speaking=True)
        assert result is True, "'yeah okay but wait' should interrupt due to 'but' and 'wait'"
