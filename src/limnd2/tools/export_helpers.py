from __future__ import annotations

import json
from pathlib import Path

from limnd2.export import ExportTarget


def terminal_progress_callback(
    current: int,
    total: int,
    file: ExportTarget,
    message: str,
) -> None:
    """
    Print a human-readable export progress message to the terminal.

    This helper is intended for callers that want a simple stdout-based
    callback implementation without coupling directly to the export internals.
    """
    if file is None:
        print(message, flush=True)
        return

    print(message, flush=True)


def json_progress_callback(
    current: int,
    total: int,
    file: ExportTarget,
    message: str,
) -> None:
    """
    Emit export progress as JSON lines matching the former export JSON mode.

    When ``file`` is provided, the callback prints:

    ``{"progress": current, "total": total, "file": "<path>"}``

    Final completion events with ``file=None`` are intentionally ignored so the
    output shape stays compatible with the old per-file JSON progress stream.
    """
    if file is None:
        return

    print(
        json.dumps(
            {
                "progress": current,
                "total": total,
                "file": str(file),
            }
        ),
        flush=True,
    )


def make_legacy_json_progress_callback() -> callable:
    """
    Return a callback that reproduces the old CLI JSON progress stream.

    The returned callback ignores the final completion event and reports only
    per-file progress using a file-count-based ``progress/total`` pair, which
    matches the former ``--progress-to-json`` CLI output contract.
    """
    state = {
        "files_done": 0,
        "file_total": None,
        "last_signature": None,
    }

    def callback(
        current: int,
        total: int,
        file: ExportTarget,
        message: str,
    ) -> None:
        if file is None:
            return

        signature = (str(file), current, total)
        if signature == state["last_signature"]:
            return
        state["last_signature"] = signature

        state["files_done"] += 1
        if state["file_total"] is None:
            state["file_total"] = max(1, total)

        print(
            json.dumps(
                {
                    "progress": state["files_done"],
                    "total": state["file_total"],
                    "file": str(file),
                }
            ),
            flush=True,
        )

    return callback
