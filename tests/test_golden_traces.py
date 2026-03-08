"""
Level 1: Golden trace comparison tests.
Runs both interpreters on available benchmarks and asserts identical output.
"""
import pytest
import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import swoosh_interp
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="Native interpreter not built")


class TestGoldenTraces:
    """Compare Python and native interpreter outputs on real benchmarks."""

    def test_both_engines_match(self, benchmark_data):
        """Run both engines on the same input and assert identical results."""
        from interpreter.runner import run_both

        name = benchmark_data['name']
        program = benchmark_data['program']
        inputs = benchmark_data['program_inputs']
        input_name = benchmark_data['input_name']

        print(f"\nTesting benchmark: {name} / {input_name}")

        explored_blocks, _ = run_both(
            program, inputs, name, input_name,
            full_trace=False, no_read_trace=False,
            extra_data=inputs.extra_data,
        )

        assert len(explored_blocks) > 0, f"No blocks explored for {name}"
        print(f"  {len(explored_blocks)} blocks explored — PASS")

    def test_explored_blocks_count(self, benchmark_data):
        """Verify we explore a reasonable number of blocks."""
        from interpreter.runner import run_python

        name = benchmark_data['name']
        program = benchmark_data['program']
        inputs = benchmark_data['program_inputs']
        input_name = benchmark_data['input_name']

        explored, _ = run_python(program, inputs, name, input_name)
        # Every benchmark should explore at least 10 blocks
        assert len(explored) >= 10, f"Only {len(explored)} blocks explored for {name}"
