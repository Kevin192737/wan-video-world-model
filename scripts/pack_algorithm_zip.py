#!/usr/bin/env python3
"""Stage Wan2.2 (no checkpoints, no release dataset) and write ../Wan2.2-no-weights.zip."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
OUT_ZIP = PROJECT_PARENT / "Wan2.2-no-weights.zip"
STAGE_NAME = "Wan2.2"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="wan22_pack_") as tmp:
        stage = Path(tmp) / STAGE_NAME
        stage.mkdir(parents=True)
        rsync = [
            "rsync",
            "-a",
            "--exclude",
            "action_head_runs",
            "--exclude",
            "release",
            "--exclude",
            ".venv",
            "--exclude",
            ".git",
            f"{REPO_ROOT}/",
            f"{stage}/",
        ]
        subprocess.run(rsync, check=True)
        if OUT_ZIP.exists():
            OUT_ZIP.unlink()
        subprocess.run(
            ["zip", "-rq", str(OUT_ZIP), STAGE_NAME],
            cwd=tmp,
            check=True,
        )
    size = OUT_ZIP.stat().st_size
    print(f"Wrote {OUT_ZIP} ({size} bytes, {size / (1024 * 1024):.2f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
