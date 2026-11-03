"""The checks the configs run in ``__post_init__``."""

from collections.abc import Sequence


def check_choice(name: str, value: str, allowed: Sequence[str]) -> None:
    """Raise ``ValueError`` unless ``value`` is one of ``allowed``."""
    if value not in allowed:
        choices = ", ".join(repr(a) for a in allowed)
        raise ValueError(f"{name} must be one of {choices}, not {value!r}")
