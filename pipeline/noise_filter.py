from .types import RawComment


class NoiseFilter:
    MIN_WORDS = 10
    MIN_SCORE = -5
    BOT_PATTERNS = ["i am a bot", "automoderator", "this action was performed automatically"]

    def run(self, comments: list[RawComment]) -> list[RawComment]:
        return [c for c in comments if self._is_valid(c)]

    def _is_valid(self, comment: RawComment) -> bool:
        if len(comment.text.split()) < self.MIN_WORDS:
            return False
        if comment.score < self.MIN_SCORE:
            return False
        text_lower = comment.text.lower()
        if any(pattern in text_lower for pattern in self.BOT_PATTERNS):
            return False
        return True
