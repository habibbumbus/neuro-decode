"""Check that installation exposes the package outside the source checkout."""

import subprocess
import sys
from pathlib import Path


def test_installed_package_imports_outside_checkout(tmp_path: Path) -> None:
    """Catch missing package files without relying on the working directory."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import neurodecode; print(neurodecode.__name__)",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "neurodecode"
