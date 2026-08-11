"""Contact-carrier retargeting primitives for the Episode-49 v16 audit.

The package is trajectory-length agnostic.  Semantic boundaries are supplied
through :class:`aloha_magsafe_semantics.schema.SemanticTimeline`; no episode
frame is assigned a meaning in this package.
"""

from .carrier import (
    LeftCarrierCandidate,
    RightCarrierCandidate,
    build_left_pinch_carrier,
    build_right_hook_carrier,
    search_left_common_rigid_carrier,
    search_right_hook_anchor,
)

__all__ = [
    "LeftCarrierCandidate",
    "RightCarrierCandidate",
    "build_left_pinch_carrier",
    "build_right_hook_carrier",
    "search_left_common_rigid_carrier",
    "search_right_hook_anchor",
]
