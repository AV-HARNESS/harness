"""R-REPO: nothing sensitive is committed; examples stay inert."""
import re
import subprocess
from pathlib import Path

from conftest import covers

ROOT = Path(__file__).parent.parent


def tracked():
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.split() if r.returncode == 0 else [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


@covers("R-REPO-01")
def test_gitignore_covers_secrets():
    gi = (ROOT / ".gitignore").read_text().split()
    for entry in (".env", ".harness/", "token.json"):
        assert entry in gi, f"{entry} missing from .gitignore"


@covers("R-REPO-01")
def test_no_secret_files_tracked():
    bad = [f for f in tracked() if re.search(r"(^|/)(\.env|token\.json|credentials.*|.*\.(pem|key|pfx|p12))$", f)
           and not f.endswith((".example", ".sample", ".template"))]
    assert bad == []
    assert not any(f.startswith(".harness/") or "/.harness/" in f for f in tracked())


@covers("R-REPO-02")
def test_example_files_hold_no_secrets():
    from harness import redact
    for name in (".env.example", "models.example.json"):
        text = (ROOT / name).read_text()
        live = [l for l in text.splitlines() if not l.lstrip().startswith("#") and "=" in l and l.split("=", 1)[1].strip()]
        if name == ".env.example":
            assert live == [], f"uncommented values in {name}: {live}"
        assert redact.mask(text) == text, f"{name} contains something that looks like a secret"
