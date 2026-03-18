"""
Level 1: Golden trace comparison tests.
Runs both interpreters on available benchmarks and asserts identical output.
"""
import pytest
import sys
import struct
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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


class TestBinaryTraceFormat:
    """Test the streaming binary trace format (.bin.zst) round-trip."""

    def test_binary_roundtrip(self, tmp_path):
        """Write a small trace dict to binary format and read it back."""
        import zstandard as zstd
        from interpreter.runner import write_trace_binary

        # Build a small trace with pre-pickled values
        trace = {
            'pc_values': {
                'positive_examples_$x_5': {pickle.dumps((42, "")), pickle.dumps((99, ""))},
                'positive_examples_$y_10': {pickle.dumps((0, ""))},
            },
            'block_values': {
                'positive_examples_$x_$bb0': {pickle.dumps((42, ""))},
            },
            'op_values': {
                'positive_examples_to_op_type_$x_5_W': {pickle.dumps((42, ""))},
            },
            'pc_registry': {
                'positive_examples_to_pc_$x': {pickle.dumps(5), pickle.dumps(10)},
            },
            'block_registry': {
                'positive_examples_to_block_$x': {pickle.dumps("$bb0")},
            },
            'total': 5,
        }

        bin_path = tmp_path / "test.trace.compact.bin.zst"
        write_trace_binary(bin_path, trace)
        assert bin_path.exists()
        assert bin_path.stat().st_size > 0

        # Read back and verify
        dctx = zstd.ZstdDecompressor()
        with open(bin_path, 'rb') as fh:
            reader = dctx.stream_reader(fh)

            def read_exact(n):
                chunks = []
                remaining = n
                while remaining > 0:
                    chunk = reader.read(remaining)
                    assert chunk, "Unexpected EOF"
                    chunks.append(chunk)
                    remaining -= len(chunk)
                return b''.join(chunks) if len(chunks) > 1 else chunks[0]

            magic = read_exact(4)
            assert magic == b"SWTR"
            version, total = struct.unpack('<BQ', read_exact(9))
            assert version == 1
            assert total == 5

            all_sections = {}
            for _ in range(5):
                cat_id, num_keys = struct.unpack('<BI', read_exact(5))
                section = {}
                for _ in range(num_keys):
                    key_len, = struct.unpack('<H', read_exact(2))
                    key = read_exact(key_len).decode()
                    num_members, = struct.unpack('<I', read_exact(4))
                    members = set()
                    for _ in range(num_members):
                        m_len, = struct.unpack('<H', read_exact(2))
                        members.add(read_exact(m_len))
                    section[key] = members
                all_sections[cat_id] = section

            footer = read_exact(4)
            assert footer == b"DONE"

        # Verify contents match
        section_map = {0: 'pc_values', 1: 'block_values', 2: 'op_values',
                       3: 'pc_registry', 4: 'block_registry'}
        for cat_id, section_name in section_map.items():
            expected = trace[section_name]
            actual = all_sections[cat_id]
            assert set(actual.keys()) == set(expected.keys()), \
                f"Key mismatch in {section_name}: {set(actual.keys())} != {set(expected.keys())}"
            for key in expected:
                assert actual[key] == expected[key], \
                    f"Value mismatch in {section_name}[{key}]"
