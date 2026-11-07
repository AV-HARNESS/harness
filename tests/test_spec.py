"""R-REPO-03: SPEC.md and the tests cannot drift apart."""
import re
from pathlib import Path

from conftest import covers

ROOT = Path(__file__).parent.parent
SPEC = (ROOT / "SPEC.md").read_text()
ID = re.compile(r"\*\*(R-[A-Z]+-\d+)\*\*([^\n]*)")


def _spec_ids():
    return {m.group(1): "[manual]" in m.group(2) for m in ID.finditer(SPEC)}


def _tested_ids():
    found = set()
    for f in (ROOT / "tests").glob("test_*.py"):
        for m in re.finditer(r"@covers\(([^)]*)\)", f.read_text()):
            found |= set(re.findall(r"R-[A-Z]+-\d+", m.group(1)))
    return found


@covers("R-REPO-03")
def test_every_automatable_requirement_has_a_test():
    missing = sorted(i for i, manual in _spec_ids().items() if not manual and i not in _tested_ids())
    assert missing == [], f"SPEC.md requirements with no test: {missing}"


@covers("R-REPO-03")
def test_tests_only_reference_real_requirements():
    unknown = sorted(_tested_ids() - set(_spec_ids()))
    assert unknown == [], f"tests reference IDs not in SPEC.md: {unknown}"


@covers("R-REPO-03")
def test_manual_items_are_not_pretending_to_be_tested():
    sneaky = sorted(i for i, manual in _spec_ids().items() if manual and i in _tested_ids())
    assert sneaky == [], f"[manual] items with tests: drop the [manual] marker: {sneaky}"


def test__spec_ids_are_unique():
    ids = [m.group(1) for m in ID.finditer(SPEC)]
    assert len(ids) == len(set(ids)) and len(ids) > 30
