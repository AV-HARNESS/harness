"""Thin shim — all logic lives in harness/providers/. Import from here for backwards compatibility."""
from .providers import generate, _text, _lower_types  # noqa: F401
from .providers.gemini import gemini_mode, gemini_alias  # noqa: F401
