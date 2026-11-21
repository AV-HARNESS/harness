"""`python -m harness bench`: how long does a model call take when the harness is not involved?

Sends a tiny prompt and a ~20k-token prompt to each model a few times. If even the tiny one is slow, the time is the
gateway/model, not the harness. A real run costs roughly (number of steps) x (tiny-call time) before any thinking."""
import statistics
import time

from . import config, llm

TINY = [{"role": "user", "parts": [{"text": "Reply with the single word: ok"}]}]
BIG = [{"role": "user", "parts": [{"text": ("The quick brown fox jumps over the lazy dog. " * 450) + "\n\nReply with the single word: ok"}]}]  # about 20k tokens


def _time(alias: str, contents: list, rounds: int) -> list[float]:
    out = []
    for _ in range(rounds):
        t0 = time.monotonic()
        llm.generate(alias, contents, "", None, effort="low")
        out.append(time.monotonic() - t0)
    return out


def _fmt(xs: list[float]) -> str:
    return f"fastest {min(xs):.1f}s, median {statistics.median(xs):.1f}s, slowest {max(xs):.1f}s"


def run(aliases: list[str], rounds: int = 5, say=print) -> int:
    bad = 0
    for alias in aliases:
        spec = config.get_model(alias) or {}
        say(f"{alias} ({spec.get('model', '?')}):")
        try:
            small = _time(alias, TINY, rounds)
            big = _time(alias, BIG, max(2, rounds // 2))
        except Exception as e:
            say(f"  failed: {str(e)[:200]}")
            bad += 1
            continue
        say(f"  tiny prompt:      {_fmt(small)}")
        say(f"  ~20k-token prompt: {_fmt(big)}")
        floor, extra = statistics.median(small), statistics.median(big) - statistics.median(small)
        say(f"  so one step costs at least {floor:.1f}s before any thinking; a 20k prompt adds about {max(extra, 0):.1f}s per step.")
        say(f"  a 30-step run is about {30 * floor:.0f}s of pure waiting" + (" (gateway/model, not the harness)." if floor >= 1.5 else "."))
    return 1 if bad else 0
