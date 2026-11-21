"""R-CLI / R-VER."""
import subprocess
import sys

from harness import cli, config, verify
from conftest import covers
from helpers import call, gem, text


def run_cli(*args, stdin=None):
    import os
    env = {**os.environ, "HARNESS_HOME": str(config.HOME), "PYTHONPATH": str(__import__("pathlib").Path(__file__).parent.parent / "src")}
    return subprocess.run([sys.executable, "-m", "harness", *args], input=stdin, capture_output=True, text=True, env=env, timeout=60)


@covers("R-CLI-01")
def test_redact_subcommand_args_and_stdin():
    r = run_cli("redact", "mail bob@example.com now")
    assert r.stdout.strip() == "mail [REDACTED:email] now" and "masked 1 item" in r.stderr
    r = run_cli("redact", stdin="password = hunter2hunter2")
    assert "hunter2" not in r.stdout and "password" in r.stdout


@covers("R-CLI-02")
def test_projects_subcommand():
    r = run_cli("projects")
    assert r.returncode == 0 and "default" in r.stdout


def test_cli_imports_without_readline(monkeypatch):
    # Windows has no readline module; importing the CLI must not fail there.
    import importlib
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "readline":
            raise ImportError("no readline on windows")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    importlib.reload(cli)


@covers("R-VER-01")
def test_verify_passes_with_working_model(fake, model, capsys, monkeypatch):
    monkeypatch.setattr(config, "MODELS", {"fake": config.MODELS["fake"]})
    fake.push((200, gem(text("pong"))), (200, gem(call("echo", text="hello-harness"))), (200, gem(text("It returned hello-harness"))))
    assert verify.main(["fake"]) == 0
    out = capsys.readouterr().out
    assert "PASS  plain call" in out and "PASS  tool-call round trip" in out and "ALL CHECKS PASSED" in out


@covers("R-VER-01")
def test_verify_fails_with_hint_and_resets_between_runs(fake, model, capsys, monkeypatch):
    monkeypatch.setattr(config, "MODELS", {"fake": config.MODELS["fake"]})
    fake.push((403, "PERMISSION_DENIED nope"))
    assert verify.main(["fake"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  plain call" in out and "Vertex AI User" in out
    fake.push((200, gem(text("pong"))), (200, gem(text("no tool call here"))))
    assert verify.main(["fake"]) == 1 and "did not call the tool" in capsys.readouterr().out
    fake.push((200, gem(text("pong"))), (200, gem(call("echo", text="x"))), (200, gem(text("ok"))))
    assert verify.main(["fake"]) == 0  # flag from earlier failures must not leak


@covers("R-VER-01")
def test_verify_unknown_alias(capsys):
    assert verify.main(["nope"]) == 1 and "unknown alias" in capsys.readouterr().out


@covers("R-VER-01")
def test_every_hint_is_reachable_text():
    assert all(k and v for k, v in verify.HINTS)


@covers("R-CLI-03")
def test_cli_opens_project_for_current_folder(tmp_path, monkeypatch):
    from harness import store
    a = tmp_path / "app"
    (a / "src").mkdir(parents=True)
    (tmp_path / "other").mkdir()
    pa = store.create_project("App", str(a))
    monkeypatch.chdir(a)
    assert cli.project_for_cwd()["id"] == pa["id"]
    monkeypatch.chdir(a / "src")  # inside a project folder -> same project
    assert cli.project_for_cwd()["id"] == pa["id"]
    n = len(store.list_projects())
    monkeypatch.chdir(tmp_path / "other")  # unknown folder -> new project named after it
    new = cli.project_for_cwd()
    assert new["name"] == "Other" and new["root"] == str((tmp_path / "other").resolve()) and len(store.list_projects()) == n + 1
    assert cli.project_for_cwd()["id"] == new["id"]  # and it is reused next time


@covers("R-CLI-04")
def test_bench_prints_timings_per_model(fake, model, capsys):
    from harness import bench
    for _ in range(8):
        fake.push((200, gem(text("ok"))))
    lines = []
    assert bench.run([model], rounds=2, say=lines.append) == 0
    out = "\n".join(lines)
    assert "tiny prompt:" in out and "median" in out and "20k-token prompt" in out and "30-step run" in out
    assert len(fake.requests) == 4  # 2 tiny + 2 big


@covers("R-CLI-04")
def test_bench_reports_a_failing_model(fake, model):
    from harness import bench
    lines = []
    fake.push((400, {"error": "bad"}))
    assert bench.run([model], rounds=1, say=lines.append) == 1 and "failed" in "\n".join(lines)
