import re

_BLANK_LINE_PATTERN = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_SPACE_PATTERN = re.compile(r"[ \t]{2,}")


def normalize_text(text: str) -> str:
    """Clean extraction artifacts while preserving paragraph boundaries."""
    normalized = text.replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in normalized.split("\n"):
        cleaned_line = _SPACE_PATTERN.sub(" ", line).strip()
        lines.append(cleaned_line)

    normalized = "\n".join(lines)
    normalized = _BLANK_LINE_PATTERN.sub("\n\n", normalized)
    return normalized.strip()
