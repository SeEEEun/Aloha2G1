"""Episode-independent ALOHA MagSafe semantic event detection.

The public API is intentionally action-domain first.  Reference timelines are
not accepted by :func:`detect_magsafe_semantics` and therefore cannot affect
candidate extraction or sequence decoding.
"""

from .detector import detect_magsafe_semantics
from .event_names import OPTIONAL_EVENTS, REQUIRED_EVENTS
from .io import load_trajectory
from .schema import EventRecord, SemanticTimeline

__all__ = [
    "EventRecord",
    "SemanticTimeline",
    "REQUIRED_EVENTS",
    "OPTIONAL_EVENTS",
    "detect_magsafe_semantics",
    "load_trajectory",
]
