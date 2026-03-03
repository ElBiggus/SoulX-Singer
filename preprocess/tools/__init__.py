"""Preprocess tools.

This package provides a thin, stable import surface for common preprocess components.

Examples:
    from preprocess.tools import (
        F0Extractor,
        PitchExtractor,
        VocalDetectionModel,
        VocalSeparationModel,
        VocalExtractionModel,
        NoteTranscriptionModel,
        LyricTranscriptionModel,
    )

Note:
    Keep these imports lightweight. If a tool pulls heavy dependencies at import time,
    consider switching to lazy imports.
"""

from __future__ import annotations

import sys

# Core tools
from .f0_extraction import F0Extractor
from .vocal_detection import VocalDetector

# Some tools may live outside this package in different layouts across branches.
# Keep the public surface stable while avoiding hard import failures.
try:
    from .vocal_separation.model import VocalSeparator  # type: ignore
except Exception:  # pragma: no cover
    VocalSeparator = None  # type: ignore

    VOCAL_SEPARATOR_IMPORT_ERROR = sys.exc_info()[1]
else:
    VOCAL_SEPARATOR_IMPORT_ERROR = None

try:
    from .note_transcription.model import NoteTranscriber  # type: ignore
except Exception:  # pragma: no cover
    NoteTranscriber = None  # type: ignore
    NOTE_TRANSCRIBER_IMPORT_ERROR = sys.exc_info()[1]
else:
    NOTE_TRANSCRIBER_IMPORT_ERROR = None
try:
    from .lyric_transcription import LyricTranscriber
except Exception:  # pragma: no cover
    LyricTranscriber = None  # type: ignore
    LYRIC_TRANSCRIBER_IMPORT_ERROR = sys.exc_info()[1]
else:
    LYRIC_TRANSCRIBER_IMPORT_ERROR = None

__all__ = [
    "F0Extractor",
    "VocalDetector",
]

if VocalSeparator is not None:
    __all__.append("VocalSeparator")
if LyricTranscriber is not None:
    __all__.append("LyricTranscriber")
if NoteTranscriber is not None:
    __all__.append("NoteTranscriber")
