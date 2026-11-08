"""R-TOOL: what the model can do."""
import pytest

from harness import config, store, tools
from conftest import covers
from pathlib import Path


@covers("R-TOOL-01")
def test_bash_runs_in_root_and_reports(inproject):
    out = tools.run("bash", {"command": "pwd; echo err >&2; exit 3"})
    assert inproject["root"] in out and "exit=3" in out and "[stderr]" in out and "err" in out


@covers("R-TOOL-01")
def test_bash_timeout(inproject):
    assert "TIMEOUT after 1s" in tools.run("bash", {"command": "sleep 5", "timeout": 1})


@covers("R-TOOL-02")
def test_relative_paths_resolve_to_root_and_dirs_created(inproject):
    root = inproject["root"]
    assert "wrote" in tools.run("write_file", {"path": "sub/dir/a.txt", "content": "l1\nl2\nl3"})
    assert open(f"{root}/sub/dir/a.txt").read() == "l1\nl2\nl3"
    assert tools.run("read_file", {"path": "sub/dir/a.txt", "offset": 2, "limit": 1}) == "2\tl2"
    assert tools.run("list_dir", {}) == "sub/"
    assert tools.run("list_dir", {"path": "sub/dir"}) == "a.txt"
    assert tools.run("read_file", {"path": f"{root}/sub/dir/a.txt"}).startswith("1\tl1")  # absolute works


@covers("R-TOOL-03")
def test_edit_file_rules(inproject):
    root = inproject["root"]
    open(f"{root}/f.txt", "w").write("a b a")
    assert "ERROR" in tools.run("edit_file", {"path": "f.txt", "old": "zzz", "new": "y"})
    assert "appears 2 times" in tools.run("edit_file", {"path": "f.txt", "old": "a", "new": "y"})
    assert open(f"{root}/f.txt").read() == "a b a"
    assert "edited" in tools.run("edit_file", {"path": "f.txt", "old": "b", "new": "B"})
    assert "2 replacement" in tools.run("edit_file", {"path": "f.txt", "old": "a", "new": "x", "replace_all": True})
    assert open(f"{root}/f.txt").read() == "x B x"


@covers("R-TOOL-04")
def test_errors_returned_not_raised(inproject):
    assert tools.run("nope", {}).startswith("ERROR: unknown tool")
    assert tools.run("read_file", {"path": "missing.txt"}).startswith("ERROR: FileNotFoundError")
    assert tools.run("read_file", {}).startswith("ERROR: TypeError")  # missing arg
    assert tools.run("read_block", {"name": "ghost"}).startswith("ERROR")


@covers("R-TOOL-05")
def test_block_tools(inproject):
    assert tools.run("list_blocks", {}) == "(no blocks yet)"
    tools.run("write_block", {"name": "plan", "content": "A"})
    tools.run("write_block", {"name": "plan", "content": "B", "append": True})
    assert tools.run("read_block", {"name": "plan"}) == "A\nB"
    assert "plan (3 chars)" in tools.run("list_blocks", {})
    assert store.read_block(inproject["id"], "plan") == "A\nB"


@covers("R-TOOL-05")
def test_block_tools_need_project():
    assert "no active project" in tools.run("list_blocks", {})


@covers("R-TOOL-06")
def test_web_fetch_html_and_text(fake, inproject):
    fake.push((200, "<html><head><style>x{}</style><script>evil()</script></head><body><h1>Title</h1><p>Body text</p>"
                    "<noscript>no</noscript></body></html>", "text/html"), (200, "plain {json}", "text/plain"))
    out = tools.run("web_fetch", {"url": fake.url + "/p"})
    assert out.startswith("HTTP 200") and "Title" in out and "Body text" in out
    assert "evil" not in out and "x{}" not in out and "no" not in out.split("Body text")[1]
    assert tools.run("web_fetch", {"url": fake.url + "/q"}) == "HTTP 200\nplain {json}"


@covers("R-TOOL-07")
def test_web_search_uses_gemini_and_lists_sources(fake, model, monkeypatch, inproject):
    monkeypatch.setattr("harness.llm.gemini_alias", lambda: "fake")
    fake.push((200, {"candidates": [{"content": {"parts": [{"text": "The answer"}, {"text": "x", "thought": True}]},
                                     "groundingMetadata": {"groundingChunks": [{"web": {"title": "T", "uri": "http://u"}}, {}]}}]}))
    out = tools.run("web_search", {"query": "q"})
    assert out == "The answer\n\nSources:\n- T: http://u"
    assert fake.requests[0]["body"]["tools"] == [{"googleSearch": {}}]


@covers("R-TOOL-07")
def test_web_search_needs_gemini(monkeypatch, inproject):
    monkeypatch.setattr(config, "MODELS", {"o": {"provider": "openai", "model": "m"}})
    assert "needs a Gemini model" in tools.run("web_search", {"query": "q"})


@covers("R-TOOL-08")
def test_long_output_is_clipped_in_the_middle(inproject):
    out = tools.run("bash", {"command": "python3 -c \"print('A'*30000 + 'Z'*30000)\""})
    assert len(out) < 21000 and "chars truncated" in out and "A" in out[:200] and out.rstrip().endswith("Z")


def test_every_registered_tool_is_declared():
    names = {d["name"] for d in tools.DECLARATIONS[0]["functionDeclarations"]}
    assert names == set(tools.REGISTRY) and len(names) == 10


@covers("R-TOOL-09")
def test_file_tools_confined_to_project_folder(inproject, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope")
    root = inproject["root"]
    import os
    os.symlink(outside, f"{root}/link")
    for args in ({"path": "../outside/secret.txt"}, {"path": str(outside / "secret.txt")}, {"path": "link/secret.txt"}):
        assert "outside the project folder" in tools.run("read_file", args)
    assert "outside the project folder" in tools.run("write_file", {"path": "../evil.txt", "content": "x"})
    assert not (tmp_path / "evil.txt").exists() and not (outside / "evil.txt").exists()
    assert "outside the project folder" in tools.run("write_file", {"path": "link/evil.txt", "content": "x"})
    assert "outside the project folder" in tools.run("list_dir", {"path": ".."})
    assert "outside the project folder" in tools.run("edit_file", {"path": "../outside/secret.txt", "old": "nope", "new": "x"})
    assert (outside / "secret.txt").read_text() == "nope"
    assert "wrote" in tools.run("write_file", {"path": "inside.txt", "content": "ok"})  # inside still works
    assert "ok" in tools.run("read_file", {"path": f"{root}/inside.txt"})


@covers("R-TOOL-09")
def test_confinement_can_be_switched_off(inproject, tmp_path, monkeypatch):
    (tmp_path / "elsewhere.txt").write_text("hi")
    monkeypatch.setattr(config, "CONFINE", False)
    assert "hi" in tools.run("read_file", {"path": str(tmp_path / "elsewhere.txt")})


@covers("R-TOOL-10")
def test_files_are_utf8_regardless_of_os_default(inproject, monkeypatch):
    project = inproject
    import locale
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a, **k: "cp1252")
    txt = "status \U0001f534 done ✓ é"
    tools.run("write_file", dict(path="u.md", content=txt))
    assert (Path(project["root"]) / "u.md").read_bytes() == txt.encode("utf-8")
    assert txt in tools.run("read_file", dict(path="u.md"))
    tools.run("edit_file", dict(path="u.md", old="done", new="ok \U0001f7e2"))
    assert "\U0001f7e2" in tools.run("read_file", dict(path="u.md"))
