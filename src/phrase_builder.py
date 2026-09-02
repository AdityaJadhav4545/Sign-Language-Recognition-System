"""
Phrase Builder
==============
Accumulates individual sign predictions into a coherent phrase.

Strategy:
  - A sign is "committed" only if it remains stable for `hold_frames`
    consecutive frames — prevents flicker from noisy predictions.
  - Space detection: if no hand is detected for `space_frames` frames,
    a word boundary is inserted.
  - Backspace: pop the last committed token.
  - Clear: reset everything.
"""

from collections import deque
import time


class PhraseBuilder:
    """
    Stateful buffer that converts per-frame sign predictions
    into a stable sequence of signs/words.

    Args:
        hold_frames:  frames a sign must be stable before committing (default 12)
        space_frames: consecutive empty frames before inserting a word space (default 20)
        max_tokens:   maximum tokens to display (rolling window)
    """

    def __init__(self,
                 hold_frames:  int = 12,
                 space_frames: int = 20,
                 max_tokens:   int = 20):
        self.hold_frames  = hold_frames
        self.space_frames = space_frames
        self.max_tokens   = max_tokens

        # Rolling buffer of the most recent predictions
        self._buf: deque = deque(maxlen=hold_frames)
        # Committed token list
        self._tokens: list[str] = []
        # Frames with no hand detected
        self._empty_count: int = 0
        # Has a space already been inserted for this gap?
        self._space_inserted: bool = False

        self._start_time = time.time()

    # ──────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────

    def push(self, sign: str | None):
        """
        Feed one frame's prediction.

        Args:
            sign: predicted class string, or None if no hand detected.
        """
        if sign is None:
            self._empty_count += 1
            self._buf.clear()   # break continuity

            # Insert space after a sustained gap (word boundary)
            if (self._empty_count >= self.space_frames
                    and not self._space_inserted
                    and self._tokens
                    and self._tokens[-1] != " "):
                self._tokens.append(" ")
                self._space_inserted = True
        else:
            self._empty_count    = 0
            self._space_inserted = False
            self._buf.append(sign)

            # Commit if buffer is full and all the same sign
            if (len(self._buf) == self.hold_frames
                    and len(set(self._buf)) == 1):
                self._commit(sign)
                self._buf.clear()

    def backspace(self):
        """Remove the last committed token."""
        if self._tokens:
            self._tokens.pop()

    def clear(self):
        """Reset phrase and buffers."""
        self._tokens.clear()
        self._buf.clear()
        self._empty_count    = 0
        self._space_inserted = False

    # ──────────────────────────────────────────
    # Output
    # ──────────────────────────────────────────

    def get_phrase(self) -> str:
        """
        Returns the current phrase as a string.
        Uses rolling window of max_tokens.
        """
        tokens = self._tokens[-self.max_tokens:]
        return "".join(tokens)

    def get_tokens(self) -> list[str]:
        return list(self._tokens[-self.max_tokens:])

    def current_candidate(self) -> str | None:
        """The sign currently in the stability buffer (not yet committed)."""
        if self._buf:
            return self._buf[-1]
        return None

    def stability_pct(self) -> float:
        """Fraction of current buffer filled with the same sign (0–1)."""
        if not self._buf:
            return 0.0
        top = max(set(self._buf), key=self._buf.count)
        return self._buf.count(top) / self.hold_frames

    # ──────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────

    def _commit(self, sign: str):
        """Append a sign only if it differs from the last token."""
        last = self._tokens[-1] if self._tokens else None
        if sign != last and sign != " ":
            self._tokens.append(sign)


# ──────────────────────────────────────────────
# Word-mode phrase builder (for dynamic model)
# ──────────────────────────────────────────────

class WordPhraseBuilder:
    """
    Same debounce logic but for whole-word predictions
    from the DynamicGestureNet.

    Committed words are space-separated.
    """

    def __init__(self, hold_sequences: int = 3, max_words: int = 30):
        self.hold = hold_sequences
        self.max  = max_words
        self._buf:   deque = deque(maxlen=hold_sequences)
        self._words: list[str] = []

    def push(self, word: str | None):
        if word is None:
            self._buf.clear()
            return
        self._buf.append(word)
        if len(self._buf) == self.hold and len(set(self._buf)) == 1:
            last = self._words[-1] if self._words else None
            if word != last:
                self._words.append(word)
            self._buf.clear()

    def get_phrase(self) -> str:
        return " ".join(self._words[-self.max:])

    def backspace(self):
        if self._words:
            self._words.pop()

    def clear(self):
        self._words.clear()
        self._buf.clear()


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    pb = PhraseBuilder(hold_frames=4, space_frames=6)

    # Simulate signing H-E-L-L-O
    for sign, frames in [("H",4), ("E",4), (None,6), ("L",4), ("L",4), ("O",4)]:
        for _ in range(frames):
            pb.push(sign)

    print(f"Phrase : '{pb.get_phrase()}'")
    print(f"Tokens : {pb.get_tokens()}")
