"""Path-protection primitive extracted for immutable experiment inputs.

Modified for LacunAI: no state configuration, discovery, wipe or restore adapters.
"""

from __future__ import annotations

import contextlib
import fnmatch
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_FOLD_CASE = sys.platform in ("win32", "darwin")


def _norm(path: Path | str) -> str:
    """Comparison key: case-folded where the filesystem is, NFC-normalized.

    `Path.resolve()` does NOT canonicalize case, so a keep entry spelled
    `~/projects` must still protect on-disk `~/Projects`. macOS also hands
    back NFD from the filesystem while a config file usually carries NFC.
    """
    text = unicodedata.normalize("NFC", str(path))
    text = os.path.normcase(text)  # also flips \\ to / on Windows
    return text.casefold() if _FOLD_CASE else text


def _spellings(path: Path) -> tuple[str, ...]:
    """Every way this path can legitimately be written: as given, and resolved.

    A keep entry may be a symlink whose destination is what the walker sees,
    or the reverse -- comparing only one spelling silently voids protection.
    """
    keys = {_norm(path)}
    with contextlib.suppress(OSError, RuntimeError):  # broken link / resolve loop
        keys.add(_norm(path.resolve()))
    return tuple(keys)


def _pattern_hits(path: Path, pattern: str) -> bool:
    """True if *path* or any ancestor matches the (absolute) glob *pattern*."""
    pat = _norm(pattern)
    for candidate in (path, *path.parents):
        for spelling in _spellings(candidate):
            if fnmatch.fnmatch(spelling, pat):
                return True
    return False


@dataclass(frozen=True)
class KeepList:
    """Paths that are protected from experimental output writes.

    Matching is deliberately generous: a path is protected if EITHER its
    literal spelling or its resolved spelling matches a keep entry in either
    of ITS spellings, compared case-insensitively where the filesystem is.
    The comparison favors preserving input artifacts.
    """

    paths: tuple[Path, ...] = ()
    #: absolute glob patterns (POSIX separators), e.g. ``~/.claude/projects/*/memory``
    #: after expansion. Patterns protect matches created at ANY time, not just
    #: ones that existed when the config was loaded.
    patterns: tuple[str, ...] = ()

    def _keys(self) -> set[str]:
        return {k for p in self.paths for k in _spellings(p)}

    def protects(self, path: Path) -> bool:
        """True if *path* is a keep entry, lives under one, or matches a pattern."""
        keys = self._keys()
        for candidate in (path, *path.parents):
            if keys.intersection(_spellings(candidate)):
                return True
        return any(_pattern_hits(path, pat) for pat in self.patterns)

    def shelters(self, path: Path) -> bool:
        """True if *path* contains a keep entry (so it must remain protected)."""
        here = set(_spellings(path))
        for kept in self.paths:
            for spelling in _spellings(kept):
                if spelling in here:
                    return True
                for parent in Path(spelling).parents:
                    if _norm(parent) in here:
                        return True
        return False
