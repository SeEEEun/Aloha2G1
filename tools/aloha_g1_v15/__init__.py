"""Generic semantic-driven orientation and Dex3 integration utilities.

The package contains no episode-specific semantic frame constants.  Episode
wrappers must provide a :class:`SemanticTimeline` and immutable Cartesian
position targets explicitly.
"""

from .semantic_input import load_human_reviewed_development_timeline

__all__ = ["load_human_reviewed_development_timeline"]
