"""The ci secret scan catches every credential shape we know of, never echoes it, and the repository is clean.

The fake credentials below are assembled at run time from harmless pieces, so this file itself never contains one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("secret_scan", ROOT / ".github" / "secret_scan.py")
secret_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(secret_scan)

ALNUM = "Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4z" * 2


def _fake(prefix: str, length: int, alphabet: str = ALNUM) -> str:
    return prefix + alphabet[:length]


FAKES = {
    "Hugging Face token": "token = " + _fake("hf" + "_", 34),
    "OpenAI-style API key": "client(key=" + _fake("sk" + "-", 40) + ")",
    "GitHub token": _fake("gh" + "p_", 36),
    "AWS access key id": "AK" + "IA" + "ABCDEFGHIJKLMNOP",
    "Slack token": _fake("xo" + "xb-", 24),
    "Google API key": _fake("AI" + "za", 35),
    "private key block": "-----BEGIN " + "RSA PRIVATE" + " KEY-----",
    "Kaggle credentials": '{"username": "someone", "key": "' + "0123456789abcdef" * 2 + '"}',
    "key in an env line": "LLM_API_KEY" + "_2=" + _fake("", 24),
    "secret assigned in code": "api_key" + ' = "' + _fake("", 24) + '"',
}


@pytest.mark.parametrize("rule", sorted(FAKES))
def test_every_rule_catches_its_shape(rule):
    assert rule in secret_scan.scan_line(FAKES[rule])


def test_a_hit_names_the_place_and_never_the_value():
    line = FAKES["Hugging Face token"]
    (hit,) = list(secret_scan.scan_lines("src/x.py", [(7, line)]))
    assert hit == "src/x.py:7: Hugging Face token"
    assert line.split()[-1] not in hit


@pytest.mark.parametrize(
    "harmless",
    [
        "LLM_API_KEY=",
        "LLM_API_KEY=<your key>",
        "key = os.environ['LLM_API_KEY']",
        'passphrase = getpass.getpass("passphrase of the final images: ")',
        "def load_keys(env=os.environ) -> list[str]:",
        "SCHEMA_MODES = ('response_format', 'guided_json', 'none')",
    ],
)
def test_ordinary_code_about_keys_is_not_flagged(harmless):
    assert secret_scan.scan_line(harmless) == []


def test_the_allow_marker_silences_a_documented_fake():
    assert secret_scan.scan_line(FAKES["GitHub token"] + "  # secret-scan: allow") == []


@pytest.mark.parametrize(
    "path",
    [".env", "src/.env.local", "kaggle.json", "conf/credentials-gcp.json", "certs/server.pem", "deploy.key",
     "home/id_ed25519", ".netrc"],
)
def test_credential_files_are_refused_by_name(path):
    assert secret_scan.scan_path_name(path)


@pytest.mark.parametrize("path", [".env.example", "src/galaxus/keys.py", "tests/test_keys.py", "keys.md"])
def test_ordinary_files_pass_by_name(path):
    assert secret_scan.scan_path_name(path) == []


def test_added_lines_are_read_from_a_unified_diff():
    diff = "\n".join(
        ["+++ b/src/a.py", "@@ -0,0 +3,2 @@", "+first", "+" + FAKES["GitHub token"], "+++ b/b.txt", "@@ -1 +1 @@", "+x"]
    )
    assert list(secret_scan.added_lines(diff))[:2] == [("src/a.py", 3, "first"), ("src/a.py", 4, FAKES["GitHub token"])]


def test_the_repository_itself_is_clean():
    assert secret_scan.scan_tree(ROOT) == []
