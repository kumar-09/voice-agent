"""
Intelligent interruption filter for voice agents.

This module provides context-aware filtering of user speech to distinguish between
passive acknowledgements (e.g., "yeah", "ok", "hmm") and active interruptions
(e.g., "stop", "wait", "no") based on whether the agent is currently speaking.

Key Features:
- Configurable ignore list for filler/acknowledgement words
- Configurable interrupt keywords that always trigger interruption
- State-based filtering (only filters when agent is speaking)
- Mixed input detection (e.g., "yeah wait" interrupts due to "wait")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Default list of words to ignore when the agent is speaking
# These are passive acknowledgements that indicate the user is listening
DEFAULT_IGNORE_WORDS: frozenset[str] = frozenset(
    [
        "yeah",
        "yes",
        "yep",
        "yup",
        "ya",
        "ok",
        "okay",
        "hmm",
        "hm",
        "mhm",
        "uh-huh",
        "uh huh",
        "uhuh",
        "right",
        "aha",
        "ah",
        "oh",
        "i see",
        "sure",
        "got it",
        "gotcha",
        "alright",
        "cool",
        "nice",
        "great",
        "good",
        "fine",
        "true",
        "indeed",
        "absolutely",
        "exactly",
        "certainly",
        "definitely",
        "of course",
        "mm",
        "mmm",
        "mmhmm",
        "mm-hmm",
    ]
)

# Default list of keywords that should always trigger an interruption
# These indicate the user wants to actively interrupt the agent
DEFAULT_INTERRUPT_KEYWORDS: frozenset[str] = frozenset(
    [
        "stop",
        "wait",
        "hold on",
        "hold up",
        "pause",
        "no",
        "nope",
        "actually",
        "but",
        "however",
        "excuse me",
        "sorry",
        "hang on",
        "one moment",
        "one second",
        "just a moment",
        "just a second",
        "let me",
        "i have",
        "i need",
        "i want",
        "can you",
        "could you",
        "would you",
        "what about",
        "what if",
        "how about",
        "listen",
        "hey",
        "hello",
        "hi",
        "question",
        "wait a minute",
        "wait a second",
        "not",
        "never",
        "don't",
        "can't",
        "won't",
    ]
)


@dataclass
class InterruptionFilterConfig:
    """Configuration for the interruption filter.

    Attributes:
        ignore_words: Set of words/phrases to ignore when agent is speaking.
            These are passive acknowledgements like "yeah", "ok", "hmm".
        interrupt_keywords: Set of keywords that always trigger an interruption,
            even when mixed with ignore words. Examples: "stop", "wait", "no".
        enabled: Whether the filter is enabled. Default True.
    """

    ignore_words: frozenset[str] = field(default_factory=lambda: DEFAULT_IGNORE_WORDS)
    interrupt_keywords: frozenset[str] = field(default_factory=lambda: DEFAULT_INTERRUPT_KEYWORDS)
    enabled: bool = True


class InterruptionFilter:
    """Filters user speech to distinguish between acknowledgements and interruptions.

    This filter is context-aware: it only filters speech when the agent is actively
    speaking. When the agent is silent, all user input is treated as valid.

    Usage:
        filter = InterruptionFilter()
        # When agent is speaking, check if we should interrupt
        if filter.should_interrupt("yeah ok", agent_is_speaking=True):
            # This returns False - it's just an acknowledgement
            pass
        if filter.should_interrupt("wait a second", agent_is_speaking=True):
            # This returns True - user wants to interrupt
            pass
        if filter.should_interrupt("yeah", agent_is_speaking=False):
            # This returns True - agent is silent, treat as valid input
            pass
    """

    def __init__(self, config: InterruptionFilterConfig | None = None) -> None:
        """Initialize the interruption filter.

        Args:
            config: Configuration for the filter. Uses defaults if not provided.
        """
        self._config = config or InterruptionFilterConfig()

    @property
    def config(self) -> InterruptionFilterConfig:
        """Get the current filter configuration."""
        return self._config

    def update_config(self, config: InterruptionFilterConfig) -> None:
        """Update the filter configuration.

        Args:
            config: New configuration to use.
        """
        self._config = config

    def should_interrupt(self, transcript: str, *, agent_is_speaking: bool) -> bool:
        """Determine if user speech should trigger an interruption.

        Args:
            transcript: The user's speech transcript.
            agent_is_speaking: Whether the agent is currently speaking.

        Returns:
            True if the speech should interrupt the agent, False otherwise.

        Logic:
            - If filter is disabled, always return True (allow interruption)
            - If agent is NOT speaking, always return True (valid input)
            - If agent IS speaking:
                - Check if transcript contains any interrupt keywords -> True
                - Check if transcript only contains ignore words -> False
                - Otherwise (contains non-ignore, non-interrupt words) -> True
        """
        if not self._config.enabled:
            return True

        if not agent_is_speaking:
            # Agent is silent, treat all input as valid
            return True

        if not transcript or not transcript.strip():
            return False

        # Normalize the transcript for comparison
        normalized = self._normalize_text(transcript)

        # Check for interrupt keywords first (these always trigger interruption)
        if self._contains_interrupt_keyword(normalized):
            return True

        # Check if the transcript only contains ignore words
        if self._is_only_ignore_words(normalized):
            return False

        # Contains other words - should interrupt
        return True

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        Converts to lowercase and removes extra whitespace.
        """
        return " ".join(text.lower().split())

    def _contains_interrupt_keyword(self, normalized_text: str) -> bool:
        """Check if text contains any interrupt keywords."""
        for keyword in self._config.interrupt_keywords:
            # Use word boundary matching for single words
            # For phrases, just check if they're in the text
            if " " in keyword:
                if keyword in normalized_text:
                    return True
            else:
                # Word boundary check using regex
                pattern = r"\b" + re.escape(keyword) + r"\b"
                if re.search(pattern, normalized_text):
                    return True
        return False

    def _is_only_ignore_words(self, normalized_text: str) -> bool:
        """Check if text consists only of words from the ignore list.

        This handles:
        - Single words: "yeah" -> True
        - Multiple ignore words: "yeah ok hmm" -> True
        - Phrases in ignore list: "i see" -> True
        - Mixed with non-ignore words: "yeah what" -> False
        """
        remaining = normalized_text

        # First, try to match and remove phrases (longer patterns first)
        # Use word boundary regex to avoid partial matches (e.g., "i see" shouldn't match "i seen")
        phrase_patterns = sorted(
            [w for w in self._config.ignore_words if " " in w], key=len, reverse=True
        )

        for phrase in phrase_patterns:
            # Use word boundary regex for phrase matching
            pattern = r"\b" + re.escape(phrase) + r"\b"
            remaining = re.sub(pattern, " ", remaining)

        # Now check if remaining words are all in ignore list
        remaining_words = remaining.split()

        if not remaining_words:
            # All text was matched by phrases
            return True

        for word in remaining_words:
            # Clean punctuation from word
            clean_word = re.sub(r"[^\w\s-]", "", word).strip()
            if clean_word and clean_word not in self._config.ignore_words:
                return False

        return True
