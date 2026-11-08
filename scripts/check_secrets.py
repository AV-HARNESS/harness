"""Fail if files about to be committed/pushed contain secrets or private terms.

Usage: python scripts/check_secrets.py [--staged]      (default: every tracked file)
- Secrets: the harness's own redaction patterns (keys, tokens, URL credentials, ...).
- Private terms: one per line in ~/.harness-denylist (or $HARNESS_DENYLIST). That file is NOT in the repo,
  so your org's hostnames, usernames etc. never need to be written down in a public place.
Test fixtures (tests/) hold fake secrets on purpose and are skipped.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HARNESS_HOME", str(ROOT / ".harness"))
from harness import redact  # noqa: E402

PLAIN_URL = re.compile(r"https?://[^\s\"'@]*")  # URLs without credentials are not secrets (TOKEN_URI = \"https://...\")
SKIP = ("tests/", "SPEC.md")
BLOCKED_NAMES = (".env", "token.json", "models.json")


def files(staged: bool) -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged else ["git", "ls-files"]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.split("\n")


def denylist() -> list[str]:
    f = Path(os.environ.get("HARNESS_DENYLIST", Path.home() / ".harness-denylist"))
    return [t.strip().lower() for t in f.read_text(encoding="utf-8").splitlines() if t.strip() and not t.startswith("#")] if f.exists() else []


def scan(paths: list[str], terms: list[str]) -> list[str]:
    problems = []
    for rel in filter(None, paths):
        if Path(rel).name in BLOCKED_NAMES:
            problems.append(f"{rel}: this kind of file must never be committed")
            continue
        if rel.startswith(SKIP) or not (ROOT / rel).is_file():
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            clean = PLAIN_URL.sub("URL", line)
            if redact.mask(clean) != clean:
                problems.append(f"{rel}:{n}: looks like a secret")
            low = line.lower()
            problems += [f"{rel}:{n}: contains a private term from your denylist" for t in terms if t in low]
    return problems


if __name__ == "__main__":
    bad = scan(files("--staged" in sys.argv), denylist())
    if bad:
        print("[check-secrets] BLOCKED:\n  " + "\n  ".join(dict.fromkeys(bad)))
        sys.exit(1)
    print("[check-secrets] ok")
