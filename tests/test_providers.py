"""R-PROV: provider wire formats, retries, errors."""
import json

import pytest

from harness import config, providers
from harness.providers import anthropic, openai
from conftest import covers
from helpers import call, gem, text

HIST = [{"role": "user", "parts": [{"text": "hi"}]}]
DECL = [{"functionDeclarations": [{"name": "t", "description": "d", "parameters": {
    "type": "OBJECT", "properties": {"a": {"type": "STRING"}}, "required": ["a"]}}]}]


@covers("R-PROV-01")
def test_gemini_api_key_request(fake, model):
    fake.push((200, gem(text("hello"), tokens=42)))
    content, usage, _ = providers.generate(model, HIST, "SYS", DECL)
    r = fake.requests[0]
    assert r["path"] == "/v1beta/models/m:generateContent" and r["headers"]["x-goog-api-key"] == "fake-key-123456789"
    assert r["body"]["systemInstruction"]["parts"][0]["text"] == "SYS" and r["body"]["tools"] == DECL
    assert content["role"] == "model" and content["parts"] == [{"text": "hello"}] and usage["promptTokenCount"] == 42


@covers("R-PROV-02")
def test_vertex_url_bearer_and_gateway(fake, monkeypatch):
    monkeypatch.setattr(config, "ACCESS_TOKEN", "tok-abc")
    monkeypatch.setattr(config, "PROJECT", "proj1")
    monkeypatch.setattr(config, "HOST", fake.url)
    monkeypatch.setitem(config.MODELS, "v", {"provider": "gemini", "model": "gm", "location": "eu1",
                                              "headers": {"X-Team": "t1"}})
    fake.push((200, gem(text("ok"))))
    providers.generate("v", HIST)
    r = fake.requests[0]
    assert r["path"] == "/v1/projects/proj1/locations/eu1/publishers/google/models/gm:generateContent"
    assert r["headers"]["authorization"] == "Bearer tok-abc" and r["headers"]["x-team"] == "t1"


@covers("R-PROV-02")
def test_vertex_default_hosts_and_missing_project(monkeypatch):
    from harness.providers import gemini
    seen = []
    monkeypatch.setattr(gemini, "_post", lambda url, h, b, r: seen.append(url) or gem(text("x")))
    monkeypatch.setattr(config, "ACCESS_TOKEN", "t")
    monkeypatch.setattr(config, "PROJECT", "p")
    monkeypatch.setitem(config.MODELS, "g", {"provider": "gemini", "model": "m"})
    providers.generate("g", HIST)
    monkeypatch.setitem(config.MODELS, "g", {"provider": "gemini", "model": "m", "location": "us-central1"})
    providers.generate("g", HIST)
    assert seen[0].startswith("https://aiplatform.googleapis.com/v1/projects/p/locations/global")
    assert seen[1].startswith("https://us-central1-aiplatform.googleapis.com/")
    monkeypatch.setattr(config, "PROJECT", "")
    with pytest.raises(RuntimeError, match="VERTEX_PROJECT"):
        providers.generate("g", HIST)


@covers("R-PROV-01")
def test_gemini_api_key_mode_detection(monkeypatch):
    from harness.providers.gemini import gemini_mode
    assert gemini_mode({"api_key_env": "X"}) == "api_key"
    assert gemini_mode({}) == "vertex"
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert gemini_mode({}) == "api_key"
    monkeypatch.setattr(config, "PROJECT", "p")
    assert gemini_mode({}) == "vertex"


@covers("R-PROV-01")
def test_gemini_missing_key_error(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    monkeypatch.setitem(config.MODELS, "k", {"provider": "gemini", "model": "m", "api_key_env": "NOPE_KEY"})
    with pytest.raises(RuntimeError, match="NOPE_KEY is not set"):
        providers.generate("k", HIST)


@covers("R-PROV-03")
def test_retries_on_429_then_succeeds(fake, model):
    fake.push((429, "slow down"), (503, "busy"), (200, gem(text("finally"))))
    content, _, _ = providers.generate(model, HIST)
    assert content["parts"][0]["text"] == "finally" and len(fake.requests) == 3


@covers("R-PROV-03")
def test_retries_exhausted_and_non_retryable(fake, model):
    fake.push(*[(500, "boom")] * 3)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        providers.generate(model, HIST, retries=2)
    assert len(fake.requests) == 3
    fake.requests.clear()
    fake.push((403, "denied"))
    with pytest.raises(RuntimeError, match="HTTP 403.*denied"):
        providers.generate(model, HIST)
    assert len(fake.requests) == 1  # not retried


@covers("R-PROV-06")
def test_empty_and_missing_output(fake, model):
    fake.push((200, {"candidates": [{"content": {}, "finishReason": "SAFETY"}]}))
    c, _, _ = providers.generate(model, HIST)
    assert "empty response" in c["parts"][0]["text"] and "SAFETY" in c["parts"][0]["text"]
    fake.push((200, {"promptFeedback": {}}))
    with pytest.raises(RuntimeError, match="No candidates"):
        providers.generate(model, HIST)


@covers("R-PROV-07")
def test_unknown_alias_and_provider(monkeypatch):
    with pytest.raises(RuntimeError, match="Unknown model 'zzz'.*flash"):
        providers.generate("zzz", HIST)
    monkeypatch.setitem(config.MODELS, "bad", {"provider": "wat", "model": "m"})
    with pytest.raises(RuntimeError, match="Unknown provider"):
        providers.generate("bad", HIST)


@covers("R-PROV-08")
def test_same_history_works_on_every_provider(fake, monkeypatch):
    monkeypatch.setenv("T_KEY", "kkkk-not-secret-shape")
    hist = [{"role": "user", "parts": [{"text": "list"}]},
            {"role": "model", "parts": [{"text": "ok"}, {"functionCall": {"name": "t", "args": {"a": "1"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "t", "response": {"output": "res"}}}]}]
    for prov, base in (("gemini", "/v1beta"), ("openai", ""), ("anthropic", "")):
        monkeypatch.setitem(config.MODELS, "x", {"provider": prov, "model": "m", "api_key_env": "T_KEY",
                                                  "base_url": fake.url + base})
        fake.push((200, {"candidates": [{"content": {"parts": [{"text": "fine"}]}}],
                         "choices": [{"message": {"content": "fine"}}], "content": [{"type": "text", "text": "fine"}]}))
        c, _, _ = providers.generate("x", hist, "S", DECL)
        assert c["parts"][0]["text"] == "fine"


@covers("R-PROV-04")
def test_openai_translation_roundtrip(fake, monkeypatch):
    monkeypatch.setenv("OAI_KEY", "sk-test-not-real-0000")
    monkeypatch.setitem(config.MODELS, "o", {"provider": "openai", "model": "gpt-x", "api_key_env": "OAI_KEY",
                                              "base_url": fake.url + "/v1", "extra": {"temperature": 0}})
    hist = [{"role": "user", "parts": [{"text": "go"}]},
            {"role": "model", "parts": [{"text": "calling"}, {"functionCall": {"name": "t", "args": {"a": "1"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "t", "response": {"output": "res"}}}]}]
    fake.push((200, {"choices": [{"message": {"content": "hi", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "t", "arguments": json.dumps({"a": "2"})}},
        {"id": "y", "type": "function", "function": {"name": "t", "arguments": "{not json"}}]}}],
        "usage": {"prompt_tokens": 7}}))
    c, usage, _ = providers.generate("o", hist, "SYS", DECL)
    r = fake.requests[0]
    assert r["path"] == "/v1/chat/completions" and r["headers"]["authorization"] == "Bearer sk-test-not-real-0000"
    b = r["body"]
    assert b["temperature"] == 0 and b["model"] == "gpt-x"
    assert [m["role"] for m in b["messages"]] == ["system", "user", "assistant", "tool"]
    assert b["messages"][2]["tool_calls"][0]["id"] == b["messages"][3]["tool_call_id"] == "call_1_0"
    assert b["tools"][0]["function"]["parameters"]["properties"]["a"]["type"] == "string"  # types lowercased
    assert c["parts"] == [{"text": "hi"}, {"functionCall": {"name": "t", "args": {"a": "2"}}},
                          {"functionCall": {"name": "t", "args": {}}}]
    assert usage == {"promptTokenCount": 7}


@covers("R-PROV-04")
def test_openai_no_key_needed_for_local_and_missing_env_errors(fake, monkeypatch):
    monkeypatch.setitem(config.MODELS, "local", {"provider": "openai", "model": "llama", "base_url": fake.url})
    fake.push((200, {"choices": [{"message": {"content": "yo"}}]}))
    providers.generate("local", HIST)
    assert "authorization" not in fake.requests[0]["headers"]
    monkeypatch.delenv("MISSING_KEY", raising=False)
    monkeypatch.setitem(config.MODELS, "o2", {"provider": "openai", "model": "m", "api_key_env": "MISSING_KEY"})
    with pytest.raises(RuntimeError, match="MISSING_KEY is not set"):
        providers.generate("o2", HIST)
    fake.push((200, {"choices": []}))
    assert "empty response" in providers.generate("local", HIST)[0]["parts"][0]["text"]


@covers("R-PROV-05")
def test_anthropic_translation_and_role_merge(fake, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "ant-not-a-real-key-0000")
    monkeypatch.setitem(config.MODELS, "a", {"provider": "anthropic", "model": "claude-x", "api_key_env": "ANT_KEY",
                                              "base_url": fake.url, "max_tokens": 99})
    hist = [{"role": "user", "parts": [{"text": "go"}]},
            {"role": "model", "parts": [{"text": "calling"}, {"functionCall": {"name": "t", "args": {"a": "1"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "t", "response": {"output": "res"}}}]},
            {"role": "user", "parts": [{"text": "and then?"}]}]
    fake.push((200, {"content": [{"type": "text", "text": "hey"}, {"type": "tool_use", "id": "z", "name": "t", "input": {"a": "3"}}],
                     "usage": {"input_tokens": 5, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2}}))
    c, usage, _ = providers.generate("a", hist, "SYS", DECL)
    r = fake.requests[0]
    assert r["path"] == "/v1/messages" and r["headers"]["x-api-key"] == "ant-not-a-real-key-0000"
    assert r["headers"]["anthropic-version"]
    b = r["body"]
    assert b["max_tokens"] == 99 and b["system"] == "SYS"
    assert [m["role"] for m in b["messages"]] == ["user", "assistant", "user"]  # consecutive user turns merged
    assert [x["type"] for x in b["messages"][2]["content"]] == ["tool_result", "text"]
    assert b["messages"][2]["content"][0]["tool_use_id"] == b["messages"][1]["content"][1]["id"]
    assert b["tools"][0]["input_schema"]["type"] == "object"
    assert c["parts"] == [{"text": "hey"}, {"functionCall": {"name": "t", "args": {"a": "3"}}}]
    assert usage["promptTokenCount"] == 10


@covers("R-PROV-05")
def test_anthropic_requires_key_env(monkeypatch):
    monkeypatch.setitem(config.MODELS, "a", {"provider": "anthropic", "model": "m"})
    with pytest.raises(RuntimeError, match="api_key_env is required"):
        providers.generate("a", HIST)


@covers("R-PROV-08")
def test_thought_signatures_stay_verbatim(fake, model):
    part = {"text": "t", "thought": True, "thoughtSignature": "SIG==", "extra": 1}
    fake.push((200, gem(part, call("x"))))
    c, _, _ = providers.generate(model, HIST)
    assert c["parts"][0] == part
    # and they are sent back unchanged on the next request
    fake.push((200, gem(text("ok"))))
    providers.generate(model, HIST + [c])
    assert fake.requests[1]["body"]["contents"][1]["parts"][0] == part


@covers("R-PROV-09")
def test_models_file_placeholders_ignored(tmp_path, monkeypatch):
    import importlib
    f = tmp_path / "models.json"
    f.write_text(json.dumps({"_comment": {"model": "x"}, "good": {"provider": "openai", "model": "gpt"},
                             "placeholder": {"provider": "openai", "model": "REPLACE_ME"}}))
    monkeypatch.setenv("HARNESS_MODELS_FILE", str(f))
    models = config._load_models()
    assert "good" in models and "placeholder" not in models and "_comment" not in models
    assert {"flash", "pro"} <= set(models)


@covers("R-PROV-09")
def test_shipped_example_models_file_is_valid_and_placeholders_inert(monkeypatch):
    from pathlib import Path
    ex = Path(__file__).parent.parent / "models.example.json"
    raw = json.loads(ex.read_text())
    monkeypatch.setenv("HARNESS_MODELS_FILE", str(ex))
    models = config._load_models()
    for alias, spec in raw.items():
        if alias.startswith("_"):
            continue
        assert spec["provider"] in ("gemini", "openai", "anthropic")
        assert (alias in models) == (not spec["model"].startswith("REPLACE"))
        assert not any(k in spec for k in ("api_key", "token", "secret"))  # keys come from env vars only


@covers("R-PROV-10")
def test_location_is_per_model(fake, monkeypatch):
    monkeypatch.setattr(config, "ACCESS_TOKEN", "t")
    monkeypatch.setattr(config, "PROJECT", "p")
    monkeypatch.setattr(config, "HOST", fake.url)
    monkeypatch.setattr(config, "LOCATION", "us-central1")
    monkeypatch.setattr(config, "MODEL_LOCATIONS", {"pro-m": "global", "flash-m": "us"})
    monkeypatch.setitem(config.MODELS, "a", {"provider": "gemini", "model": "pro-m"})
    monkeypatch.setitem(config.MODELS, "b", {"provider": "gemini", "model": "flash-m"})
    monkeypatch.setitem(config.MODELS, "c", {"provider": "gemini", "model": "other"})
    monkeypatch.setitem(config.MODELS, "d", {"provider": "gemini", "model": "pro-m", "location": "eu1"})
    for alias in "abcd":
        fake.push((200, gem(text("ok"))))
        providers.generate(alias, HIST)
    locs = [r["path"].split("/locations/")[1].split("/")[0] for r in fake.requests]
    assert locs == ["global", "us", "us-central1", "eu1"]


@covers("R-PROV-11")
def test_source_slash_model_alias(fake, monkeypatch):
    monkeypatch.setitem(config.MODELS, "oa", {"provider": "openai", "model": "base", "base_url": fake.url, "api_key_env": ""})
    assert config.get_model("oa/gpt-x") == {"provider": "openai", "model": "gpt-x", "base_url": fake.url, "api_key_env": ""}
    assert config.get_model("nope/gpt-x") is None and config.get_model("oa") ["model"] == "base"
    fake.push((200, {"choices": [{"message": {"content": "hi"}}], "usage": {}}))
    providers.generate("oa/gpt-x", HIST)
    assert fake.requests[0]["body"]["model"] == "gpt-x"
