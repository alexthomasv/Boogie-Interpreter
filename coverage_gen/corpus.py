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


class Corpus:
    def __init__(self):
        self.entries: list[CorpusEntry] = []
        self.covered: set = set()
        self.params_line: str = ""

    def add(self, path, inputs: ProgramInputs, covered_blocks: set,
            params_line: str = "") -> bool:
        """Add an entry if it brings new coverage. Returns True if added."""
        new_blocks = covered_blocks - self.covered
        if not new_blocks and self.entries:
            return False
        # Energy = number of newly-found blocks (minimum 1)
        energy = max(1, len(new_blocks))
        entry = CorpusEntry(path=path, inputs=inputs,
                            covered_blocks=covered_blocks,
                            params_line=params_line, energy=energy)
        self.entries.append(entry)
        self.covered |= covered_blocks
        if params_line and not self.params_line:
            self.params_line = params_line
        return True

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
                self.add(p, inputs, covered, params_line=params_line)
                loaded += 1
            except Exception as exc:
                print(f"[corpus] Warning: could not load seed {p.name}: {exc}")
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
