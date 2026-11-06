"""Privacy: what is masked before a model call, and what the model may not touch."""
from pathlib import Path

import pytest

from harness import config, redact, tools
from conftest import covers


@pytest.fixture(autouse=True)
def terms(monkeypatch):
    monkeypatch.setattr(config, "REDACT_TERMS", ["acmecorp", "project-falcon"])


def m(text):
    return redact.mask(text)



@covers("R-PRIV-01")
def test_api_keys_and_tokens():
    assert "AIza" not in m("key=AIzaSyA1234567890abcdefghijklmnopqrstuv")
    assert "[REDACTED:api_key]" in m("use sk-ant-abcdefghijklmnopqrstuvwxyz123456")
    assert "[REDACTED:token]" in m("Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789")
    assert "[REDACTED:jwt]" in m("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop")
    assert "BEGIN" not in m("-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----")


@covers("R-PRIV-02")
def test_assignments_keep_name_and_hide_value():
    out = m('DB_PASSWORD = "hunter2hunter2"\nclient_secret: s3cr3tvalue99')
    assert "hunter2" not in out and "s3cr3tvalue99" not in out
    assert "DB_PASSWORD" in out and "client_secret" in out


@covers("R-PRIV-02")
def test_code_is_left_alone():
    code = 'token = tokens[0]\nmax_tokens = 8192\nTOKEN_URL = os.environ.get("OAUTH_TOKEN_URL", "")\n'
    assert m(code) == code


@covers("R-PRIV-03")
def test_url_credentials_email_ip_ssn_card():
    assert "pw123" not in m("postgres://admin:pw123@db.internal/x")
    assert m("mail bob@example.com") == "mail [REDACTED:email]"
    assert "10.4.5.6" not in m("host 10.4.5.6") and "8.8.8.8" in m("host 8.8.8.8")
    assert "123-45-6789" not in m("ssn 123-45-6789")
    assert "4111111111111111" not in m("card 4111 1111 1111 1111")
    assert "1234567890123" in m("id 1234567890123")  # fails Luhn: not a card


@covers("R-PRIV-04")
def test_known_env_secrets_and_terms(monkeypatch):
    monkeypatch.setenv("MY_API_SECRET", "plainwordsecret")
    assert m("the value is plainwordsecret ok") == "the value is [REDACTED:env:MY_API_SECRET] ok"
    assert "AcmeCorp" not in m("Deploy for AcmeCorp via project-falcon")


@covers("R-PRIV-05")
def test_request_masking_preserves_structure_and_input():
    contents = [
        {"role": "user", "parts": [{"text": "mail bob@example.com"}]},
        {"role": "model", "parts": [{"text": "thinking", "thought": True, "thoughtSignature": "AIzaSy" + "x" * 35},
                                    {"functionCall": {"name": "bash", "args": {"command": "echo bob@example.com"}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "bash", "response": {"output": "bob@example.com"}}}]},
    ]
    out, system, counts = redact.mask_request(contents, "admin@example.com")
    assert contents[0]["parts"][0]["text"] == "mail bob@example.com"  # input untouched
    assert out[0]["parts"][0]["text"] == "mail [REDACTED:email]"
    assert out[1]["parts"][0] == contents[1]["parts"][0]  # thought + signature passed through
    assert out[1]["parts"][1]["functionCall"]["args"]["command"] == "echo [REDACTED:email]"
    assert out[2]["parts"][0]["functionResponse"]["response"]["output"] == "[REDACTED:email]"
    assert system == "[REDACTED:email]" and counts["email"] == 4


@covers("R-PRIV-06", "R-PRIV-07")
def test_tool_guards(tmp_path):
    (tmp_path / ".env").write_text("A=1")
    (tmp_path / ".env.example").write_text("A=")
    (tmp_path / "x.txt").write_text("hello")
    p = tools.PROJECT.set({"id": "t", "root": str(tmp_path), "name": "t"})
    assert tools.run("read_file", {"path": ".env"}).startswith("ERROR")
    assert tools.run("read_file", {"path": "ca.pem"}).startswith("ERROR")
    assert "A=" in tools.run("read_file", {"path": ".env.example"})
    assert tools.run("write_file", {"path": "x.txt", "content": "key=[REDACTED:secret]"}).startswith("ERROR")
    assert tools.run("edit_file", {"path": "x.txt", "old": "hello", "new": "[REDACTED:email]"}).startswith("ERROR")
    assert Path(tmp_path / "x.txt").read_text() == "hello"
    tools.PROJECT.reset(p)


@covers("R-PRIV-06")
def test_harness_dir_and_custom_blocklist(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BLOCK_FILES", config.BLOCK_FILES + ["*.secret-thing"])
    (tmp_path / ".harness").mkdir()
    (tmp_path / ".harness" / "token.json").write_text("{}")
    (tmp_path / "a.secret-thing").write_text("x")
    (tmp_path / "app.sample").write_text("ok")
    p = tools.PROJECT.set({"id": "t", "root": str(tmp_path), "name": "t"})
    try:
        assert tools.run("read_file", {"path": ".harness/token.json"}).startswith("ERROR")
        assert tools.run("read_file", {"path": "a.secret-thing"}).startswith("ERROR")
        assert tools.run("write_file", {"path": "credentials.json", "content": "{}"}).startswith("ERROR")
        assert not (tmp_path / "credentials.json").exists()
        assert "ok" in tools.run("read_file", {"path": "app.sample"})
    finally:
        tools.PROJECT.reset(p)


@covers("R-PRIV-08")
def test_redaction_can_be_turned_off(monkeypatch, fake, model):
    from harness import providers
    fake.push((200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}))
    monkeypatch.setattr(config, "REDACT", False)
    providers.generate(model, [{"role": "user", "parts": [{"text": "mail bob@example.com"}]}])
    assert "bob@example.com" in str(fake.requests[-1]["body"])


@covers("R-PRIV-09")
@pytest.mark.parametrize("provider", ["gemini", "openai", "anthropic"])
def test_every_provider_sends_only_masked_text(provider, fake, monkeypatch):
    from harness import providers
    monkeypatch.setenv("T_KEY", "k-not-a-secret-pattern")
    spec = {"provider": provider, "model": "m", "api_key_env": "T_KEY",
            "base_url": fake.url + ("/v1beta" if provider == "gemini" else "")}
    monkeypatch.setitem(config.MODELS, "p", spec)
    fake.push((200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "choices": [{"message": {"content": "ok"}}],
                     "content": [{"type": "text", "text": "ok"}]}))
    providers.generate("p", [{"role": "user", "parts": [{"text": "password = hunter2hunter2 and bob@example.com"}]}],
                       system="admin@example.com")
    sent = str(fake.requests[-1]["body"])
    assert "hunter2" not in sent and "bob@example.com" not in sent and "admin@example.com" not in sent
    assert "[REDACTED:" in sent


@covers("R-PRIV-09")
def test_compaction_and_web_search_are_masked(fake, model, monkeypatch, session):
    from harness import agent, tools as t
    monkeypatch.setattr(config, "KEEP_TURNS", 1)
    session["messages"] = [
        {"role": "user", "parts": [{"text": "first bob@example.com"}]}, {"role": "model", "parts": [{"text": "a"}]},
        {"role": "user", "parts": [{"text": "second"}]}, {"role": "model", "parts": [{"text": "b"}]}]
    fake.push((200, {"candidates": [{"content": {"parts": [{"text": "summary"}]}}]}))
    assert agent.compact(session)
    assert "bob@example.com" not in str(fake.requests[-1]["body"])
    monkeypatch.setitem(config.MODELS, "fake", {**config.MODELS["fake"]})
    monkeypatch.setattr("harness.llm.gemini_alias", lambda: "fake")
    fake.push((200, {"candidates": [{"content": {"parts": [{"text": "ans"}]}}]}))
    t.web_search("what is 10.1.2.3 and bob@example.com")
    assert "bob@example.com" not in str(fake.requests[-1]["body"])
