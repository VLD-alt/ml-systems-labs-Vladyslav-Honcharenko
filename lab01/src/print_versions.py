"""Print Python and library versions for reproducibility.

Reads the pinned packages from the repo-level requirements.txt and reports
the installed version of each one next to the required version.

Usage (from repo root):
    python lab01/src/print_versions.py > lab01/results/versions.txt
"""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "requirements.txt"


def read_pins(path):
    pins = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, pinned = line.split("==", 1)
        pins.append((name.strip(), pinned.strip()))
    return pins


def same_version(installed, pinned):
    """Compare versions after PEP 440 normalisation (e.g. 0.1.1-2209 == 0.1.1.post2209)."""
    try:
        return Version(installed) == Version(pinned)
    except InvalidVersion:
        return installed == pinned


def main():
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Implementation: {platform.python_implementation()}")
    print(f"Executable  : {sys.executable}")
    print(f"Prefix      : {sys.prefix}")
    print(f"Base prefix : {sys.base_prefix}")
    isolated = sys.prefix != sys.base_prefix
    print(f"Isolated venv: {'YES' if isolated else 'NO - not running inside a virtual environment'}")
    print(f"Platform    : {platform.platform()}")
    print(f"Machine     : {platform.machine()}")
    print(f"Processor   : {platform.processor()}")
    print()
    print(f"{'package':<18} {'installed':<28} {'required':<28} status")
    print("-" * 82)

    mismatches = 0
    for name, pinned in read_pins(REQUIREMENTS):
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "NOT INSTALLED"
        status = "OK" if same_version(installed, pinned) else "MISMATCH"
        mismatches += status != "OK"
        print(f"{name:<18} {installed:<28} {pinned:<28} {status}")

    print("-" * 82)
    print(f"Python 3.11.8 required: {'OK' if sys.version.split()[0] == '3.11.8' else 'MISMATCH'}")
    print(f"Package mismatches: {mismatches}")


if __name__ == "__main__":
    main()
