#!/usr/bin/env python3
# Modified for the standalone LacunAI package; historical wire formats are preserved.
"""Run the optional research batch from a verified source bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lacunai.experimental.batch import finalize_exception, main  # noqa: E402

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--root" in sys.argv:
            try:
                location = Path(sys.argv[sys.argv.index("--root") + 1])
                finalize_exception(location, sys.argv[1], error)
            except Exception as finalization_error:
                print(
                    json.dumps({"state": "finalization_failed", "error": str(finalization_error)}),
                    file=sys.stderr,
                    flush=True,
                )
        print(
            json.dumps({"state": "failed", "type": type(error).__name__, "error": str(error)}),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from error
