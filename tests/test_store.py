"""R-STORE: projects, blocks, sessions on disk."""
import json

import pytest

from harness import config, store
from conftest import covers


@covers("R-STORE-01")
def test_unique_slugs_and_root_must_exist(tmp_path):
    a = store.create_project("My Project!", str(tmp_path))
    b = store.create_project("My Project!", str(tmp_path))
    assert a["id"] == "my-project" and b["id"] == "my-project-2"
    assert store.get_project("my-project")["root"] == str(tmp_path.resolve())
    with pytest.raises(ValueError, match="does not exist"):
        store.create_project("x", str(tmp_path / "missing"))
    assert [p["id"] for p in store.list_projects()] == ["my-project", "my-project-2"]


@covers("R-STORE-02")
def test_delete_project_keeps_root(project):
    root = project["root"]
    open(f"{root}/precious.txt", "w").write("keep")
    store.write_block(project["id"], "n", "x")
    s = store.create(project["id"])
    store.delete_project(project["id"])
    assert open(f"{root}/precious.txt").read() == "keep"
    with pytest.raises(KeyError):
        store.get_project(project["id"])
    with pytest.raises(KeyError):
        store.load(s["id"])


@covers("R-STORE-03")
def test_block_crud(project):
    pid = project["id"]
    store.write_block(pid, "goals", "one")
    store.write_block(pid, "goals", "two", append=True)
    store.write_block(pid, "new", "x", append=True)  # append to missing creates
    assert store.read_block(pid, "goals") == "one\ntwo"
    assert [b["name"] for b in store.list_blocks(pid)] == ["goals", "new"]
    store.delete_block(pid, "goals")
    store.delete_block(pid, "goals")  # idempotent
    with pytest.raises(KeyError):
        store.read_block(pid, "goals")
    for bad in ("../x", "a/b", "", ".hid", "a b"):
        with pytest.raises(ValueError):
            store.write_block(pid, bad, "x")


@covers("R-STORE-04")
def test_session_roundtrip_and_order(project):
    a = store.create(project["id"], "first")
    b = store.create(project["id"], "second", model="pro", instructions="inst")
    b["messages"].append({"role": "user", "parts": [{"text": "héllo ✓", "thoughtSignature": "S"}]})
    store.save(b)
    back = store.load(b["id"])
    assert back["messages"] == b["messages"] and back["model"] == "pro" and back["instructions"] == "inst"
    assert a["model"] == project["model"]
    store.save(a)  # touch a: now newest
    assert [s["id"] for s in store.list_all(project["id"])][0] == a["id"]
    store.delete(a["id"])
    store.delete(a["id"])  # idempotent
    assert [s["id"] for s in store.list_all(project["id"])] == [b["id"]]


@covers("R-STORE-05")
def test_no_path_traversal(project):
    for bad in ("../x", "..", "a/b", "", "A"):
        with pytest.raises(KeyError):
            store.get_project(bad)
        with pytest.raises(KeyError):
            store.load(bad)
    with pytest.raises(KeyError):
        store.read_block("../..", "x")


@covers("R-STORE-06")
def test_default_project_and_legacy_migration():
    config.LEGACY_SESSIONS_DIR.mkdir()
    (config.LEGACY_SESSIONS_DIR / "abc123.json").write_text(json.dumps(
        {"id": "abc123", "title": "old", "model": "flash", "messages": [], "created": 1, "updated": 1}))
    (config.LEGACY_SESSIONS_DIR / "broken.json").write_text("{nope")
    store.ensure_default()
    assert [p["id"] for p in store.list_projects()] == ["default"]
    assert store.load("abc123")["project"] == "default"
    store.ensure_default()  # idempotent
    assert len(store.list_projects()) == 1


@covers("R-STORE-07")
def test_update_project(project, tmp_path):
    new_root = tmp_path / "other"
    new_root.mkdir()
    p = store.update_project(project["id"], name="Renamed", root=str(new_root), model="pro")
    assert p["name"] == "Renamed" and p["root"] == str(new_root.resolve()) and p["model"] == "pro"
    assert store.update_project(project["id"], model="not-a-model")["model"] == "pro"
    with pytest.raises(ValueError):
        store.update_project(project["id"], root=str(tmp_path / "nope"))
    assert store.get_project(project["id"])["id"] == project["id"]  # id never changes on rename


@covers("R-STORE-04")
def test_corrupt_files_do_not_break_listing(project):
    store.create(project["id"])
    (config.PROJECTS_DIR / project["id"] / "sessions" / "zz.json").write_text("{bad")
    (config.PROJECTS_DIR / "junk").mkdir()
    (config.PROJECTS_DIR / "junk" / "project.json").write_text("{bad")
    assert len(store.list_all(project["id"])) == 1
    assert [p["id"] for p in store.list_projects()] == [project["id"]]


@covers("R-STORE-09")
def test_everything_is_stored_in_the_central_folder_not_the_project(project):
    from pathlib import Path
    store.write_block(project["id"], "notes", "x")
    s = store.create(project["id"])
    root = Path(project["root"])
    assert list(root.iterdir()) == []  # nothing written into the user's project folder
    for p in (config.PROJECTS_DIR / project["id"] / "blocks" / "notes.md", config.PROJECTS_DIR / project["id"] / "sessions" / f"{s['id']}.json"):
        assert p.exists() and config.HOME in p.parents


@covers("R-STORE-09")
def test_harness_home_env_is_used_and_tilde_expanded(tmp_path):
    import os, subprocess, sys
    from pathlib import Path
    env = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "HARNESS_HOME": "~/central",
           "PYTHONPATH": str(Path(__file__).parent.parent / "src")}
    out = subprocess.run([sys.executable, "-c", "from harness import config; print(config.HOME)"], capture_output=True, text=True, env=env, cwd=tmp_path).stdout.strip()
    assert out == str(tmp_path / "central") and (tmp_path / "central" / "projects").is_dir()
    env.pop("HARNESS_HOME")
    out = subprocess.run([sys.executable, "-c", "from harness import config; print(config.HOME)"], capture_output=True, text=True, env=env, cwd=tmp_path).stdout.strip()
    assert out.endswith(".harness") and "src" not in out.replace("\\", "/").split("/")[-2:]
@covers("R-STORE-08")
def test_writes_are_atomic_and_leave_no_temp_files(project, monkeypatch):
    import os
    s = store.create(project["id"])
    path = config.PROJECTS_DIR / project["id"] / "sessions" / f"{s['id']}.json"
    seen = []
    real = os.replace

    def spy(a, b):  # at the moment of the swap the old file is still complete and the new one is complete too
        seen.append((json.loads(open(b).read())["title"], json.loads(open(a).read())["title"]))
        return real(a, b)
    monkeypatch.setattr(os, "replace", spy)
    s["title"] = "new title"
    store.save(s)
    assert seen and seen[-1][1] == "new title" and seen[-1][0] != "new title"
    assert json.loads(path.read_text())["title"] == "new title"
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no .tmp left behind


@covers("R-STORE-08")
def test_reader_retries_an_empty_file_then_succeeds(project, monkeypatch):
    s = store.create(project["id"])
    path = config.PROJECTS_DIR / project["id"] / "sessions" / f"{s['id']}.json"
    good = path.read_text()
    path.write_text("")  # what a reader sees while another thread is mid-write

    def heal(_):
        path.write_text(good)
    monkeypatch.setattr("time.sleep", heal)
    assert store.load(s["id"])["id"] == s["id"]


@covers("R-STORE-08")
def test_permanently_corrupt_file_still_raises(project):
    s = store.create(project["id"])
    (config.PROJECTS_DIR / project["id"] / "sessions" / f"{s['id']}.json").write_text("")
    with pytest.raises(json.JSONDecodeError):
        store.load(s["id"])
