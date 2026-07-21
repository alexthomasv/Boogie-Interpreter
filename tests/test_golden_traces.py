"""Golden trace smoke tests for the Rust interpreter."""
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import swoosh_interp
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

native_required = pytest.mark.skipif(not HAS_NATIVE, reason="Native interpreter not built")
GOLDEN_TRACE_MAX_STEPS = 100_000_000


class TestGoldenTraces:
    """Run the Rust interpreter on real benchmarks."""

    pytestmark = [
        native_required,
        pytest.mark.integration,
        pytest.mark.requires_compiled_package,
        pytest.mark.slow,
    ]

    def test_native_engine_runs(self, benchmark_data, tmp_path):
        """Run the Rust engine on the same input and assert it explores blocks."""
        from interpreter.runner import run_native

        name = benchmark_data['name']
        program = benchmark_data['program']
        inputs = benchmark_data['program_inputs']
        input_name = benchmark_data['input_name']

        print(f"\nTesting benchmark: {name} / {input_name}")

        result = run_native(
            program, inputs, input_name,
            raw_log_path=tmp_path / f"{input_name}.trace.raw.zst",
            no_trace=True,
            return_status=True,
            max_steps=GOLDEN_TRACE_MAX_STEPS,
        )
        explored_blocks = set(result.get("explored_blocks") or [])

        assert len(explored_blocks) > 0, f"No blocks explored for {name}"
        print(f"  {len(explored_blocks)} blocks explored - PASS")

    def test_explored_blocks_count(self, benchmark_data, tmp_path):
        """Verify explored block accounting is internally consistent."""
        from interpreter.runner import run_native

        name = benchmark_data['name']
        program = benchmark_data['program']
        inputs = benchmark_data['program_inputs']
        input_name = benchmark_data['input_name']

        result = run_native(
            program, inputs, input_name,
            raw_log_path=tmp_path / f"{input_name}.trace.raw.zst",
            no_trace=True,
            return_status=True,
            max_steps=GOLDEN_TRACE_MAX_STEPS,
        )
        explored = set(result.get("explored_blocks") or [])
        assert result.get("status") in {
            "ok",
            "assert_violation",
            "assume_violation",
            "step_limit",
        }
        assert len(explored) > 0, f"No blocks explored for {name}"
        assert result.get("blocks_explored") == len(explored)
