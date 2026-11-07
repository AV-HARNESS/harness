"""Builders for fake Gemini generateContent responses."""


def gem(*parts, tokens=10, **candidate_extra):
    cand = {"content": {"role": "model", "parts": list(parts)}, **candidate_extra}
    return {"candidates": [cand], "usageMetadata": {"promptTokenCount": tokens}}


def text(t):
    return {"text": t}


def call(_name, **args):
    return {"functionCall": {"name": _name, "args": args}}
