"""Coverage-guided input generator for the Boogie interpreter.

Hybrid greybox mutation (Phase 1) + constraint-guided (Phase 2, future).
Follows the QSYM pattern: mutation until coverage stalls, then targeted
constraint solving.

Usage:
    python3 -m interpreter.coverage_gen.driver test_packages/aead_pkg/ \\
        [--phase greybox|hybrid] [--iters N] [--timeout S] [--out-dir PATH]
"""

import argparse
import pickle
import random
import re
import sys
from pathlib import Path


def _next_gen_idx(out_dir: Path) -> int:
    """Find the next available gen_N index in out_dir."""
    idxs = []
    for p in out_dir.glob("gen_*.input"):
        m = re.fullmatch(r"gen_(\d+)", p.stem)
        if m:
            idxs.append(int(m.group(1)))
    return max(idxs, default=-1) + 1


def run_greybox(corpus, evaluator, out_dir: Path, iters: int) -> int:
    """Run the greybox mutation phase. Returns the number of new inputs written."""
    from interpreter.coverage_gen.mutator import mutate, splice
    from interpreter.coverage_gen.writer import write_input_file

    gen_idx = _next_gen_idx(out_dir)
    new_count = 0
    stale = 0
    stale_limit = max(50, iters // 10)

    for i in range(iters):
        entry = corpus.pick()
        if entry is None:
            break

        # 10% splice, 90% mutation
        if len(corpus.entries) > 1 and random.random() < 0.1:
            other = corpus.pick()
            candidate = splice(entry.inputs, other.inputs)
        else:
            candidate = mutate(entry.inputs)

        covered = evaluator.run(candidate, f"gen_{gen_idx}")
        new_blocks = covered - corpus.covered

        if new_blocks:
            params_line = entry.params_line or corpus.params_line
            text = write_input_file(candidate, params_line)
            out_path = out_dir / f"gen_{gen_idx}.input"
            out_path.write_text(text)
            corpus.add(out_path, candidate, covered, params_line=params_line)
            gen_idx += 1
            new_count += 1
            stale = 0
            print(f"[greybox] iter={i+1:5d}  +{len(new_blocks):3d} blocks → "
                  f"{len(corpus.covered):5d} total | wrote gen_{gen_idx-1}.input")
        else:
            stale += 1
            if stale >= stale_limit:
                print(f"[greybox] Coverage stalled for {stale} iterations — stopping")
                break

        if (i + 1) % 200 == 0:
            print(f"[greybox] iter={i+1}/{iters} | "
                  f"covered={len(corpus.covered)} blocks | "
                  f"corpus={len(corpus.entries)} entries")

    return new_count


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Coverage-guided input generator for the Boogie interpreter"
    )
    parser.add_argument("pkg_dir",
                        help="Compiled package dir (test_packages/<name>_pkg/)")
    parser.add_argument("--phase", choices=["greybox", "hybrid"],
                        default="hybrid",
                        help="Generation strategy (default: hybrid)")
    parser.add_argument("--iters", type=int, default=1000,
                        help="Greybox iteration budget (default: 1000)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Per-execution timeout in seconds (default: 30)")
    parser.add_argument("--out-dir",
                        help="Output directory for generated .input files "
                             "(default: test_input/<name>/)")
    parser.add_argument("--seed",
                        help="Seed corpus directory "
                             "(default: test_input/<name>/)")
    args = parser.parse_args(argv)

    pkg_path = Path(args.pkg_dir)
    m = re.fullmatch(r"(.+)_pkg", pkg_path.name)
    if not m:
        print(f"ERROR: {pkg_path.name!r} does not match <name>_pkg pattern",
              file=sys.stderr)
        sys.exit(1)
    test_name = m.group(1)

    # Load compiled program
    program_pkl = pkg_path / f"{test_name}.pkl"
    if not program_pkl.exists():
        print(f"ERROR: {program_pkl} not found", file=sys.stderr)
        sys.exit(1)
    print(f"[driver] Loading {test_name} ...")
    program = pickle.loads(program_pkl.read_bytes())

    seed_dir = Path(args.seed) if args.seed else Path(f"test_input/{test_name}")
    out_dir = Path(args.out_dir) if args.out_dir else seed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from interpreter.coverage_gen.corpus import Corpus
    from interpreter.coverage_gen.evaluator import Evaluator

    evaluator = Evaluator(program, test_name, timeout=args.timeout)
    corpus = Corpus()

    # Seed phase
    print(f"[driver] Loading seeds from {seed_dir} ...")
    n_seeds = corpus.load_seeds(seed_dir, evaluator)
    print(f"[driver] Loaded {n_seeds} seeds | "
          f"{len(corpus.covered)} blocks covered | "
          f"{len(corpus.entries)} corpus entries")

    if not corpus.entries:
        print("[driver] No seeds found — cannot fuzz without a starting input.",
              file=sys.stderr)
        sys.exit(1)

    # Greybox phase
    print(f"[driver] Starting greybox phase ({args.iters} iterations) ...")
    n_new = run_greybox(corpus, evaluator, out_dir, iters=args.iters)
    print(f"[driver] Greybox done: {n_new} new inputs written | "
          f"{len(corpus.covered)} total blocks covered")


if __name__ == "__main__":
    main()
