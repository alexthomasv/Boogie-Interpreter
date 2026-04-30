"""Corpus management for coverage-guided fuzzing.

Maintains a set of interesting inputs (those that added new block coverage),
with AFL-style energy weighting: inputs covering rarer blocks fuzz more.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from interpreter.utils.inputs import ProgramInputs


@dataclass
class CorpusEntry:
    path: Optional[Path]
    inputs: ProgramInputs
    covered_blocks: set
    params_line: str = ""
    energy: int = 1
    source: str = "unknown"
    interesting_reason: str = "coverage"


class Corpus:
    def __init__(self, max_trace_entries: int = 128):
        self.entries: list[CorpusEntry] = []
        self.covered: set = set()
        self.params_line: str = ""
        self.max_trace_entries = max_trace_entries
        self.trace_signatures: set = set()
        self.loaded_inputs = 0
        self.invalid_inputs = 0
        self.duplicate_inputs = 0
        self.coverage_inputs = 0
        self.trace_interesting_inputs = 0
        self.generated_trace_inputs = 0
        self.assertion_inputs = 0
        self.assertion_pcs = {}
        self.seed_assertion_feedback_inputs = 0
        self.seed_assertion_feedback_pcs = {}
        self.assertion_signatures: set = set()
        self.assertion_feedback_entries: list[CorpusEntry] = []

    def add(self, path, inputs: ProgramInputs, covered_blocks: set,
            params_line: str = "", *, source: str = "unknown",
            allow_trace: bool = False,
            interesting_reason: str = "coverage") -> bool:
        """Add an entry if it brings new coverage or useful trace diversity."""
        new_blocks = covered_blocks - self.covered
        if new_blocks or not self.entries:
            reason = interesting_reason if new_blocks else "initial"
            energy = max(1, len(new_blocks))
            self.coverage_inputs += 1
            self.trace_signatures.add(self.trace_signature(inputs, covered_blocks))
        elif allow_trace and self.should_add_trace(inputs, covered_blocks):
            reason = "trace"
            energy = 1
            self.trace_interesting_inputs += 1
            if source != "existing":
                self.generated_trace_inputs += 1
            self.trace_signatures.add(self.trace_signature(inputs, covered_blocks))
        else:
            self.duplicate_inputs += 1
            return False

        entry = CorpusEntry(path=path, inputs=inputs,
                            covered_blocks=covered_blocks,
                            params_line=params_line, energy=energy,
                            source=source, interesting_reason=reason)
        self.entries.append(entry)
        self.covered |= covered_blocks
        if params_line and not self.params_line:
            self.params_line = params_line
        return True

    def add_assertion_feedback(self, path, inputs: ProgramInputs,
                               covered_blocks: set, params_line: str = "",
                               *, source: str = "unknown") -> None:
        """Record a failing assert input for solver feedback only.

        These entries are not valid concrete inputs and must not affect normal
        coverage, trace, or fuzz corpus state.
        """
        entry = CorpusEntry(path=path, inputs=inputs,
                            covered_blocks=covered_blocks,
                            params_line=params_line, energy=1,
                            source=source,
                            interesting_reason="assertion-feedback")
        self.assertion_feedback_entries.append(entry)
        if params_line and not self.params_line:
            self.params_line = params_line

    def trace_signature(self, inputs: ProgramInputs, covered_blocks: set):
        """Stable signature for no-new-coverage traces worth keeping."""
        havoc = []
        for name, inp in sorted(inputs.variables.items()):
            if inp.havoc_seq is not None:
                havoc.append((name, tuple(inp.havoc_seq)))
        extra_len = len(inputs.extra_data or b"")
        return (tuple(sorted(covered_blocks)), tuple(havoc), extra_len)

    def should_add_trace(self, inputs: ProgramInputs, covered_blocks: set) -> bool:
        if self.trace_interesting_inputs >= self.max_trace_entries:
            return False
        sig = self.trace_signature(inputs, covered_blocks)
        return sig not in self.trace_signatures

    def pick(self) -> Optional[CorpusEntry]:
        """Pick a corpus entry weighted by energy (AFL-style power schedule)."""
        if not self.entries:
            return None
        total = sum(e.energy for e in self.entries)
        r = random.uniform(0, total)
        cumsum = 0.0
        for entry in self.entries:
            cumsum += entry.energy
            if r <= cumsum:
                return entry
        return self.entries[-1]

    def load_seeds(self, seed_dir: Path, evaluator) -> int:
        """Load all .input files from seed_dir as corpus seeds.

        Returns the number of seeds loaded.
        """
        from interpreter.utils.input_parser import parse_input_file

        seed_dir = Path(seed_dir)
        if not seed_dir.exists():
            return 0

        loaded = 0
        for p in sorted(seed_dir.glob("*.input")):
            try:
                inputs = parse_input_file(p)
                params_line = _extract_params_line(p)
                covered = evaluator.run(inputs, p.stem)
                self.add(p, inputs, covered, params_line=params_line,
                         source="existing", allow_trace=True)
                loaded += 1
            except Exception as exc:
                self.invalid_inputs += 1
                print(f"[corpus] Warning: could not load seed {p.name}: {exc}")
        self.loaded_inputs += loaded
        return loaded


def _extract_params_line(path: Path) -> str:
    """Extract the '// @params ...' line from an .input file."""
    try:
        for line in path.read_text().splitlines():
            if line.strip().startswith("// @params"):
                return line.strip()
    except Exception:
        pass
    return ""
