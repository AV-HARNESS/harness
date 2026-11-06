import os
import tempfile

os.environ["HARNESS_HOME"] = tempfile.mkdtemp()
os.environ["HARNESS_REDACT_TERMS"] = "acmecorp,project-falcon"

from pathlib import Path  # noqa: E402

from harness import redact, tools  # noqa: E402


def m(text):
    return redact.mask(text)


def test_api_keys_and_tokens():
    assert "AIza" not in m("key=AIzaSyA1234567890abcdefghijklmnopqrstuv")
    assert "[REDACTED:api_key]" in m("use sk-ant-abcdefghijklmnopqrstuvwxyz123456")
    assert "[REDACTED:token]" in m("Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789")
    assert "[REDACTED:jwt]" in m("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop")
    assert "BEGIN" not in m("-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----")


def test_assignments_keep_name_and_hide_value():
    out = m('DB_PASSWORD = "hunter2hunter2"\nclient_secret: s3cr3tvalue99')
    assert "hunter2" not in out and "s3cr3tvalue99" not in out
    assert "DB_PASSWORD" in out and "client_secret" in out


def test_code_is_left_alone():
    code = 'token = tokens[0]\nmax_tokens = 8192\nTOKEN_URL = os.environ.get("OAUTH_TOKEN_URL", "")\n'
    assert m(code) == code


def test_url_credentials_email_ip_ssn_card():
    assert "pw123" not in m("postgres://admin:pw123@db.internal/x")
    assert m("mail bob@example.com") == "mail [REDACTED:email]"
    assert "10.4.5.6" not in m("host 10.4.5.6") and "8.8.8.8" in m("host 8.8.8.8")
    assert "123-45-6789" not in m("ssn 123-45-6789")
    assert "4111111111111111" not in m("card 4111 1111 1111 1111")
    assert "1234567890123" in m("id 1234567890123")  # fails Luhn: not a card


def test_known_env_secrets_and_terms(monkeypatch):
    monkeypatch.setenv("MY_API_SECRET", "plainwordsecret")
    assert m("the value is plainwordsecret ok") == "the value is [REDACTED:env:MY_API_SECRET] ok"
    assert "AcmeCorp" not in m("Deploy for AcmeCorp via project-falcon")


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
