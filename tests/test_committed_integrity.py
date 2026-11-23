"""Git byte identity checks independent of Windows' default text encoding."""

import hashlib
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("evaluate_final")


def git_bytes(monkeypatch, runner, content):
    def run(args, **kwargs):
        assert args[-2:] == ["show", "HEAD:synthetic.py"]
        assert kwargs == {"capture_output": True, "check": True}
        return SimpleNamespace(stdout=content)

    monkeypatch.setattr(runner.subprocess, "run", run)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_utf8_git_identity_is_independent_of_cp1252(runner, monkeypatch, newline):
    authentic = "score = 'R² — café'\n"
    expected = hashlib.sha256(authentic.encode("utf-8")).hexdigest()
    raw = authentic.replace("\n", newline).encode("utf-8")
    # Reproduce the old Windows decoding explicitly on every test platform.
    old = raw.decode("cp1252").replace("\r\n", "\n").replace("\r", "\n")
    assert hashlib.sha256(old.encode("utf-8")).hexdigest() != expected
    git_bytes(monkeypatch, runner, raw)
    freeze = {"implementation_sha256_utf8_lf": {"synthetic.py": expected}}
    runner.verify_committed_implementation(freeze)
    assert runner.read_committed_utf8("synthetic.py") == authentic


def test_genuinely_altered_git_blob_is_rejected(runner, monkeypatch):
    authentic = "score = 'R² — café'\n"
    expected = hashlib.sha256(authentic.encode("utf-8")).hexdigest()
    git_bytes(monkeypatch, runner, authentic.replace("score", "wrong").encode("utf-8"))
    with pytest.raises(ValueError, match="not committed at HEAD: synthetic.py"):
        runner.verify_committed_implementation(
            {"implementation_sha256_utf8_lf": {"synthetic.py": expected}}
        )


def test_invalid_utf8_git_blob_fails_closed(runner, monkeypatch):
    git_bytes(monkeypatch, runner, b"# invalid UTF-8: \xff\n")
    with pytest.raises(UnicodeDecodeError):
        runner.read_committed_utf8("synthetic.py")


def test_git_read_failure_is_not_suppressed(runner, monkeypatch):
    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(runner.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        runner.read_committed_utf8("missing.py")


def test_freeze_json_also_uses_utf8_and_still_requires_exact_identity(
    runner, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    freeze = {
        "description": "R² — café",
        "implementation_sha256_utf8_lf": {},
        "validation_run_id": "synthetic-evidence",
    }
    raw = json.dumps(freeze, ensure_ascii=False).encode("utf-8")

    def run(args, **kwargs):
        assert args[-2:] == ["show", "HEAD:configs/final_evaluation.json"]
        assert kwargs == {"capture_output": True, "check": True}
        return SimpleNamespace(stdout=raw)

    monkeypatch.setattr(runner.subprocess, "run", run)
    evidence = Path("runs/synthetic-evidence")
    evidence.mkdir(parents=True)
    run_file = evidence / "run.json"
    run_file.write_text(
        json.dumps(
            {
                "status": "passed",
                "partition": "validation",
                "freeze_sha256": runner.canonical_sha(freeze),
            }
        ),
        encoding="utf-8",
    )
    (evidence / "verification.json").write_text(
        json.dumps({"run_json_sha256": runner.sha(run_file)}), encoding="utf-8"
    )
    runner.require_committed_freeze(freeze)
    with pytest.raises(ValueError, match="freeze is not committed exactly"):
        runner.require_committed_freeze({**freeze, "description": "altered"})
