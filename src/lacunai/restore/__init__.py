"""Sparse touched-tensor snapshots and exact checkpoint undo."""

from .checkpoints import (
    SnapshotError,
    prepare_snapshot,
    restore_snapshot,
    write_edited_checkpoint,
)

__all__ = ["SnapshotError", "prepare_snapshot", "restore_snapshot", "write_edited_checkpoint"]
