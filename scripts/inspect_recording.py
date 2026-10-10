"""Load one recording and print its raw-data summary; no feature processing."""

import json
import sys

from neurodecode.data.loaders import load_recording, summarize_recording

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/inspect_recording.py PATH_TO_RECORDING")
    print(json.dumps(summarize_recording(load_recording(sys.argv[1])), indent=2))
