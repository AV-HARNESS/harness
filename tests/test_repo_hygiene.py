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


@covers("R-REPO-04")
def test_check_secrets_script(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_secrets", ROOT / "scripts" / "check_secrets.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    assert cs.scan(tracked(), []) == []  # the repo itself is clean
    monkeypatch.setattr(cs, "ROOT", tmp_path)
    (tmp_path / "a.py").write_text("key = 'AIzaSyA1234567890abcdefghijklmnopqrstuv'\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("gateway is corp-internal.example\n", encoding="utf-8")
    (tmp_path / ".env").write_text("X=1\n", encoding="utf-8")
    out = cs.scan(["a.py", "b.md", ".env"], ["corp-internal"])
    assert any("a.py:1" in p for p in out) and any("b.md:1" in p for p in out) and any(".env" in p for p in out)
