"""Protocol-v2 entry point for Example 21 ARC-AGI-1 latent reasoning."""

from __future__ import annotations

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(
        pathlib.Path(__file__).with_name("21-latent-reasoning-in-context.py"),
        run_name="__main__",
    )
