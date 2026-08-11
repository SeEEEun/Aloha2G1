"""ALOHA MagSafe semantic detector v2.

The canonical v1 schema/API stays authoritative.  V2 only replaces evidence
generation, global decoding, and confidence attribution.
"""

from .detector import detect_magsafe_semantics

__all__ = ["detect_magsafe_semantics"]
