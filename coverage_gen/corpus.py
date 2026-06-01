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
    path_features: dict = field(default_factory=dict)
    params_line: str = ""
    energy: int = 1
    source: str = "unknown"
    interesting_reason: str = "coverage"


class Corpus:
    def __init__(self, max_trace_entries: int = 128,
                 coverage_metric: str = "block"):
        self.entries: list[CorpusEntry] = []
        self.covered: set = set()
        self.covered_edges: set = set()
        self.covered_context_edges: set = set()
        self.params_line: str = ""
        self.max_trace_entries = max_trace_entries
        self.coverage_metric = coverage_metric
        self.trace_signatures: set = set()
        self.loaded_inputs = 0
        self.invalid_inputs = 0
        self.duplicate_inputs = 0
        self.coverage_inputs = 0
        self.path_inputs = 0
        self.generated_path_inputs = 0
        self.trace_interesting_inputs = 0
        self.generated_trace_inputs = 0
        self.assertion_inputs = 0
        self.assertion_pcs = {}
        self.seed_assertion_feedback_inputs = 0
        self.seed_assertion_feedback_pcs = {}
        self.assertion_signatures: set = set()
        self.assertion_feedback_entries: list[CorpusEntry] = []
        self._feature_frequency_cache: dict[str, dict] = {}
        self._coverage_objective_cache: dict[str, set] = {}

    def add(self, path, inputs: ProgramInputs, covered_blocks: set,
            params_line: str = "", *, source: str = "unknown",
            allow_trace: bool = False,
            interesting_reason: str = "coverage",
            path_features: dict | None = None) -> bool:
        """Add an entry if it brings new coverage or useful trace diversity."""
        path_features = normalize_path_features(path_features, covered_blocks)
        new_blocks = covered_blocks - self.covered
        new_edges = set(path_features.get("edges", ())) - self.covered_edges
        new_context_edges = (
            set(path_features.get("context_edges", ()))
            - self.covered_context_edges
        )
        new_paths = self._new_paths_for_metric(new_edges, new_context_edges)

        if new_blocks or not self.entries:
            reason = interesting_reason if new_blocks else "initial"
            energy = max(1, len(new_blocks))
            self.coverage_inputs += 1
            self.trace_signatures.add(self.trace_signature(
                inputs, covered_blocks, path_features))
        elif new_paths:
            reason = self.coverage_metric
            energy = max(1, len(new_paths))
            self.path_inputs += 1
            if source != "existing":
                self.generated_path_inputs += 1
            self.trace_signatures.add(self.trace_signature(
                inputs, covered_blocks, path_features))
        elif allow_trace and self.should_add_trace(
            inputs, covered_blocks, path_features):
            reason = "trace"
            energy = 1
            self.trace_interesting_inputs += 1
            if source != "existing":
                self.generated_trace_inputs += 1
            self.trace_signatures.add(self.trace_signature(
                inputs, covered_blocks, path_features))
        else:
            self.duplicate_inputs += 1
            return False

        entry = CorpusEntry(path=path, inputs=inputs,
                            covered_blocks=covered_blocks,
                            path_features=path_features,
                            params_line=params_line, energy=energy,
                            source=source, interesting_reason=reason)
        self.entries.append(entry)
        self._invalidate_feature_cache()
        self.covered |= covered_blocks
        self.covered_edges |= set(path_features.get("edges", ()))
        self.covered_context_edges |= set(
            path_features.get("context_edges", ()))
        if params_line and not self.params_line:
            self.params_line = params_line
        return True

    def _new_paths_for_metric(self, new_edges: set,
                              new_context_edges: set) -> set:
        if self.coverage_metric == "context-edge":
            return new_context_edges
        if self.coverage_metric in {"edge", "branch"}:
            return new_edges
        return set()

    def add_assertion_feedback(self, path, inputs: ProgramInputs,
                               covered_blocks: set, params_line: str = "",
                               *, source: str = "unknown",
                               path_features: dict | None = None) -> None:
        """Record a failing assert input for solver feedback only.

        These entries are not valid concrete inputs and must not affect normal
        coverage, trace, or fuzz corpus state.
        """
        entry = CorpusEntry(path=path, inputs=inputs,
                            covered_blocks=covered_blocks,
                            path_features=normalize_path_features(
                                path_features, covered_blocks),
                            params_line=params_line, energy=1,
                            source=source,
                            interesting_reason="assertion-feedback")
        self.assertion_feedback_entries.append(entry)
        if params_line and not self.params_line:
            self.params_line = params_line

    def _invalidate_feature_cache(self) -> None:
        self._feature_frequency_cache.clear()
        self._coverage_objective_cache.clear()

    def trace_signature(self, inputs: ProgramInputs, covered_blocks: set,
                        path_features: dict | None = None):
        """Stable signature for no-new-coverage traces worth keeping."""
        path_features = normalize_path_features(path_features, covered_blocks)
        havoc = []
        for name, inp in sorted(inputs.variables.items()):
            if inp.havoc_seq is not None:
                havoc.append((name, tuple(inp.havoc_seq)))
        extra_len = len(inputs.extra_data or b"")
        sequence = tuple(path_features.get("sequence", ()))
        if not sequence:
            sequence = tuple(sorted(covered_blocks))
        return (sequence, tuple(havoc), extra_len)

    def should_add_trace(self, inputs: ProgramInputs, covered_blocks: set,
                         path_features: dict | None = None) -> bool:
        if self.trace_interesting_inputs >= self.max_trace_entries:
            return False
        sig = self.trace_signature(inputs, covered_blocks, path_features)
        return sig not in self.trace_signatures

    def pick(self) -> Optional[CorpusEntry]:
        """Pick a corpus entry weighted by energy (AFL-style power schedule)."""
        return self._pick_weighted(lambda entry: float(entry.energy))

    def pick_rare(self) -> Optional[CorpusEntry]:
        """Pick a corpus entry with extra energy for low-frequency paths."""
        return self._pick_weighted(
            lambda entry: float(entry.energy) + self.entry_rarity(entry)
        )

    def _pick_weighted(self, weight_fn) -> Optional[CorpusEntry]:
        if not self.entries:
            return None
        weights = [max(1.0, float(weight_fn(entry))) for entry in self.entries]
        total = sum(weights)
        r = random.uniform(0, total)
        cumsum = 0.0
        for entry, weight in zip(self.entries, weights):
            cumsum += weight
            if r <= cumsum:
                return entry
        return self.entries[-1]

    def feature_frequencies(self, metric: str | None = None) -> dict:
        """Count how often each retained corpus feature appears."""
        metric = metric or self.coverage_metric
        if metric in self._feature_frequency_cache:
            return dict(self._feature_frequency_cache[metric])
        counts = {}
        for entry in self.entries:
            features = coverage_features(
                entry.covered_blocks,
                entry.path_features,
                metric=metric,
            )
            for feature in features:
                counts[feature] = counts.get(feature, 0) + 1
        self._feature_frequency_cache[metric] = dict(counts)
        return counts

    def entry_rarity(self, entry: CorpusEntry,
                     metric: str | None = None) -> float:
        """AFLFast/FairFuzz-style score for entries covering rare features."""
        counts = self.feature_frequencies(metric)
        features = coverage_features(
            entry.covered_blocks,
            entry.path_features,
            metric=metric or self.coverage_metric,
        )
        return sum(1.0 / max(1, counts.get(feature, 1))
                   for feature in features)

    def coverage_objectives(self, metric: str | None = None) -> set:
        """All coverage objectives currently represented in the corpus."""
        metric = metric or self.coverage_metric
        if metric in self._coverage_objective_cache:
            return set(self._coverage_objective_cache[metric])
        objectives = set()
        for entry in self.entries:
            objectives.update(coverage_features(
                entry.covered_blocks,
                entry.path_features,
                metric=metric,
            ))
        self._coverage_objective_cache[metric] = set(objectives)
        return objectives

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


def normalize_path_features(path_features: dict | None,
                            covered_blocks: set | None = None) -> dict:
    sequence = tuple((path_features or {}).get("sequence") or ())
    if not sequence and covered_blocks:
        sequence = tuple(sorted(covered_blocks))
    edges = tuple((path_features or {}).get("edges") or _edges(sequence))
    context_edges = tuple(
        (path_features or {}).get("context_edges")
        or _context_edges(sequence)
    )
    return {
        "sequence": sequence,
        "edges": edges,
        "context_edges": context_edges,
    }


def path_features_from_sequence(sequence) -> dict:
    sequence = tuple(sequence or ())
    return {
        "sequence": sequence,
        "edges": tuple(_edges(sequence)),
        "context_edges": tuple(_context_edges(sequence)),
    }


def coverage_features(covered_blocks: set | None,
                      path_features: dict | None = None,
                      *, metric: str = "block") -> set:
    """Return weighted coverage objective labels for set-cover reduction."""
    path_features = normalize_path_features(path_features, covered_blocks)
    features = {("block", block) for block in (covered_blocks or set())}
    if metric in {"edge", "branch"}:
        features.update(("edge", edge) for edge in path_features["edges"])
    elif metric == "context-edge":
        features.update(
            ("context-edge", edge)
            for edge in path_features["context_edges"]
        )
    return features


def _edges(sequence: tuple) -> tuple:
    return tuple(zip(sequence, sequence[1:]))


def _context_edges(sequence: tuple) -> tuple:
    out = []
    for idx in range(1, len(sequence)):
        prev_prev = sequence[idx - 2] if idx >= 2 else "<entry>"
        out.append((prev_prev, sequence[idx - 1], sequence[idx]))
    return tuple(out)
