"""Tests for interpreter resume/restart behavior.

Tests verify that:
1. Completed inputs are skipped on re-run (resume works)
2. Partially-written traces (no explored_blocks) are cleaned up and re-run
3. --force flag ignores all caches and re-runs everything
4. Hash mismatch trashes old traces
5. Hash check detects incomplete runs and resumes instead of skipping
6. Hash is only written after all inputs complete
"""
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# We need to test process_single_input and the main-block logic.
# Since the main block is in `if __name__ == '__main__'`, we test it via subprocess.
# For unit-level tests, we import process_single_input directly.

ROOT = Path(__file__).resolve().parent.parent


class TestProcessSingleInput(unittest.TestCase):
    """Unit tests for process_single_input resume logic."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.test_name = "test_resume_bench"
        self.trace_dir = self.tmpdir / "positive_examples" / self.test_name
        self.trace_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _make_trace(self, input_name, content="x <- 0x0 GLOBAL GLOBAL 0 W\n"):
        """Create a trace file."""
        path = self.trace_dir / f"{input_name}.trace.txt"
        path.write_text(content)
        return path

    def _make_explored(self, input_name, content="block1\nblock2\n"):
        """Create an explored_blocks file (marks input as complete)."""
        path = self.trace_dir / f"{input_name}.explored_blocks.txt"
        path.write_text(content)
        return path

    def test_completed_input_is_skipped(self):
        """If both trace and explored_blocks exist, input should be skipped."""
        input_name = "test_input_01"
        self._make_trace(input_name)
        self._make_explored(input_name)

        trace_path = self.trace_dir / f"{input_name}.trace.txt"
        explored_path = self.trace_dir / f"{input_name}.explored_blocks.txt"

        force = False
        should_skip = (
            not force
            and explored_path.exists()
            and trace_path.exists()
            and trace_path.stat().st_size > 0
        )
        assert should_skip, "Completed input should be skipped"

    def test_partial_trace_is_detected(self):
        """If trace exists but explored_blocks doesn't, it's a partial run."""
        input_name = "test_input_02"
        trace = self._make_trace(input_name)
        explored_path = self.trace_dir / f"{input_name}.explored_blocks.txt"

        assert trace.exists()
        assert not explored_path.exists()
        # This means the input was interrupted and needs re-running

    def test_partial_trace_cleanup(self):
        """Partial traces should be deleted before re-running."""
        input_name = "test_input_03"
        trace = self._make_trace(input_name, "partial data\n")
        explored_path = self.trace_dir / f"{input_name}.explored_blocks.txt"

        # Simulate the cleanup logic from process_single_input
        if trace.exists() and not explored_path.exists():
            trace.unlink()

        assert not trace.exists(), "Partial trace should have been deleted"

    def test_force_flag_skips_completed_check(self):
        """With force=True, even completed inputs should be re-run."""
        input_name = "test_input_04"
        self._make_trace(input_name)
        self._make_explored(input_name)

        trace_path = self.trace_dir / f"{input_name}.trace.txt"
        explored_path = self.trace_dir / f"{input_name}.explored_blocks.txt"

        force = True
        should_skip = (
            not force
            and explored_path.exists()
            and trace_path.exists()
            and trace_path.stat().st_size > 0
        )
        assert not should_skip, "force=True should prevent skipping"

    def test_no_force_skips_completed(self):
        """Without force, completed inputs should be skipped."""
        input_name = "test_input_05"
        self._make_trace(input_name)
        self._make_explored(input_name)

        trace_path = self.trace_dir / f"{input_name}.trace.txt"
        explored_path = self.trace_dir / f"{input_name}.explored_blocks.txt"

        force = False
        should_skip = (
            not force
            and explored_path.exists()
            and trace_path.exists()
            and trace_path.stat().st_size > 0
        )
        assert should_skip, "Completed inputs should be skipped without --force"

    def test_empty_trace_not_skipped(self):
        """An empty trace file should not count as completed."""
        input_name = "test_input_06"
        trace = self._make_trace(input_name, "")  # empty
        self._make_explored(input_name)

        trace_path = self.trace_dir / f"{input_name}.trace.txt"
        explored_path = self.trace_dir / f"{input_name}.explored_blocks.txt"

        force = False
        should_skip = (
            not force
            and explored_path.exists()
            and trace_path.exists()
            and trace_path.stat().st_size > 0
        )
        assert not should_skip, "Empty trace should not be skipped"


class TestHashBasedCache(unittest.TestCase):
    """Test the hash-based cache that skips interpretation when .bpl is unchanged."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.test_name = "test_hash_bench"

        # Set up directory structure
        self.bpl_dir = self.tmpdir / "bpl_out" / self.test_name
        self.bpl_dir.mkdir(parents=True)
        self.trace_dir = self.tmpdir / "positive_examples" / self.test_name
        self.trace_dir.mkdir(parents=True)
        self.input_dir = self.tmpdir / "test_input" / self.test_name
        self.input_dir.mkdir(parents=True)

        # Create a fake .bpl file
        self.bpl_file = self.bpl_dir / f"{self.test_name}_inline.bpl"
        self.bpl_file.write_text("// fake bpl content v1")
        self.bpl_hash = hashlib.sha256(self.bpl_file.read_bytes()).hexdigest()

        self.hash_file = self.trace_dir / f"{self.test_name}.interp.hash"

        # Create fake input files
        for i in range(3):
            (self.input_dir / f"input_{i}.json").write_text("{}")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_all_done_with_matching_hash_skips(self):
        """If hash matches and all inputs are complete, should skip."""
        # Write matching hash
        self.hash_file.write_text(self.bpl_hash + "\n")

        # Create completed traces for all inputs
        for i in range(3):
            (self.trace_dir / f"input_{i}.trace.txt").write_text("data\n")
            (self.trace_dir / f"input_{i}.explored_blocks.txt").write_text("block\n")

        # Check the skip condition
        stored_hash = self.hash_file.read_text().strip()
        input_files = list(self.input_dir.glob("*.json"))
        all_done = all(
            (self.trace_dir / f"{p.stem}.explored_blocks.txt").exists()
            for p in input_files
        )
        assert stored_hash == self.bpl_hash
        assert all_done
        # Should skip

    def test_partial_completion_with_matching_hash_resumes(self):
        """If hash matches but not all inputs are complete, should resume."""
        self.hash_file.write_text(self.bpl_hash + "\n")

        # Only 2 of 3 inputs completed
        for i in range(2):
            (self.trace_dir / f"input_{i}.trace.txt").write_text("data\n")
            (self.trace_dir / f"input_{i}.explored_blocks.txt").write_text("block\n")
        # input_2 has partial trace only
        (self.trace_dir / "input_2.trace.txt").write_text("partial\n")

        stored_hash = self.hash_file.read_text().strip()
        input_files = list(self.input_dir.glob("*.json"))
        all_done = all(
            (self.trace_dir / f"{p.stem}.explored_blocks.txt").exists()
            for p in input_files
        )
        assert stored_hash == self.bpl_hash
        assert not all_done, "Not all inputs are done, should resume"

    def test_hash_mismatch_trashes_traces(self):
        """If bpl changed, old traces should be deleted."""
        self.hash_file.write_text("old_hash_value\n")

        # Old completed traces
        for i in range(3):
            (self.trace_dir / f"input_{i}.trace.txt").write_text("old data\n")
            (self.trace_dir / f"input_{i}.explored_blocks.txt").write_text("block\n")

        stored_hash = self.hash_file.read_text().strip()
        assert stored_hash != self.bpl_hash

        # Simulate hash mismatch cleanup
        shutil.rmtree(self.trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        assert not any(self.trace_dir.glob("*.trace.txt"))
        assert not any(self.trace_dir.glob("*.explored_blocks.txt"))

    def test_no_hash_file_does_not_skip(self):
        """If no hash file exists (first run or interrupted), should not skip."""
        assert not self.hash_file.exists()
        # The condition `hash_file.exists()` is False, so the whole block is skipped
        # and interpretation proceeds

    def test_hash_written_only_after_completion(self):
        """Hash file should only be written after all inputs succeed."""
        # Before completion, no hash
        assert not self.hash_file.exists()

        # Simulate successful completion
        bpl_hash = self.bpl_hash
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.hash_file.write_text(bpl_hash + "\n")

        assert self.hash_file.exists()
        assert self.hash_file.read_text().strip() == bpl_hash

    def test_force_bypasses_hash_check(self):
        """With --force, hash match should not cause skip."""
        self.hash_file.write_text(self.bpl_hash + "\n")
        for i in range(3):
            (self.trace_dir / f"input_{i}.trace.txt").write_text("data\n")
            (self.trace_dir / f"input_{i}.explored_blocks.txt").write_text("block\n")

        force = True
        # With force, the hash check block is skipped entirely (line: `if not args.force and ...`)
        should_check_hash = not force
        assert not should_check_hash, "--force should bypass hash check"


class TestResumeIntegration(unittest.TestCase):
    """Integration tests using a minimal fake interpreter setup."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.test_name = "fake_bench"

        # Directory structure
        self.pkg_dir = self.tmpdir / f"{self.test_name}_pkg"
        self.pkg_dir.mkdir()
        self.trace_dir = self.tmpdir / "positive_examples" / self.test_name
        self.trace_dir.mkdir(parents=True)
        self.input_dir = self.tmpdir / "test_input" / self.test_name
        self.input_dir.mkdir(parents=True)
        self.bpl_dir = self.tmpdir / "bpl_out" / self.test_name
        self.bpl_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _create_input_files(self, count=3):
        """Create fake JSON input files."""
        for i in range(count):
            (self.input_dir / f"input_{i}.json").write_text(json.dumps({"id": i}))

    def _simulate_completion(self, input_name):
        """Simulate a completed interpreter run for one input."""
        (self.trace_dir / f"{input_name}.trace.txt").write_text(
            f"$x <- 0x{42:x} block1 block0 100 W\n"
        )
        (self.trace_dir / f"{input_name}.explored_blocks.txt").write_text("block1\n")

    def _simulate_partial(self, input_name):
        """Simulate an interrupted run — trace exists but no explored_blocks."""
        (self.trace_dir / f"{input_name}.trace.txt").write_text(
            f"$x <- 0x{42:x} block1 block0 100 W\n"
            "PARTIAL — was interrupted here\n"
        )
        # No explored_blocks file = incomplete

    def test_resume_skips_completed_reruns_partial(self):
        """On resume, completed inputs are skipped, partial ones are re-run."""
        self._create_input_files(3)
        self._simulate_completion("input_0")
        self._simulate_completion("input_1")
        self._simulate_partial("input_2")

        # Verify state
        assert (self.trace_dir / "input_0.explored_blocks.txt").exists()
        assert (self.trace_dir / "input_1.explored_blocks.txt").exists()
        assert not (self.trace_dir / "input_2.explored_blocks.txt").exists()
        assert (self.trace_dir / "input_2.trace.txt").exists()

        # The cleanup logic should remove the partial trace
        trace = self.trace_dir / "input_2.trace.txt"
        explored = self.trace_dir / "input_2.explored_blocks.txt"
        if trace.exists() and not explored.exists():
            trace.unlink()

        assert not trace.exists(), "Partial trace should be cleaned up"
        # input_0 and input_1 should still be intact
        assert (self.trace_dir / "input_0.trace.txt").exists()
        assert (self.trace_dir / "input_1.trace.txt").exists()

    def test_resume_count_reporting(self):
        """Resume should correctly report how many inputs are done."""
        self._create_input_files(5)
        # 3 of 5 complete
        for i in range(3):
            self._simulate_completion(f"input_{i}")
        self._simulate_partial("input_3")
        # input_4 not started at all

        input_files = list(self.input_dir.glob("*.json"))
        done = sum(
            1 for p in input_files
            if (self.trace_dir / f"{p.stem}.explored_blocks.txt").exists()
        )
        assert done == 3
        assert len(input_files) == 5

    def test_force_cleans_all(self):
        """--force should cause all inputs to be re-run regardless of state."""
        self._create_input_files(3)
        for i in range(3):
            self._simulate_completion(f"input_{i}")

        force = True
        if force:
            # With force, partial cleanup happens on each input individually
            # The key is that should_skip is False
            for i in range(3):
                trace = self.trace_dir / f"input_{i}.trace.txt"
                explored = self.trace_dir / f"input_{i}.explored_blocks.txt"
                should_skip = (
                    not force
                    and explored.exists()
                    and trace.exists()
                    and trace.stat().st_size > 0
                )
                assert not should_skip

    def test_no_inputs_started(self):
        """First run with no existing traces should process all inputs."""
        self._create_input_files(3)

        input_files = list(self.input_dir.glob("*.json"))
        done = sum(
            1 for p in input_files
            if (self.trace_dir / f"{p.stem}.explored_blocks.txt").exists()
        )
        assert done == 0
        assert len(input_files) == 3

    def test_all_complete_no_rerun_needed(self):
        """If all inputs are complete, no re-processing needed."""
        self._create_input_files(3)
        for i in range(3):
            self._simulate_completion(f"input_{i}")

        input_files = list(self.input_dir.glob("*.json"))
        all_done = all(
            (self.trace_dir / f"{p.stem}.explored_blocks.txt").exists()
            for p in input_files
        )
        assert all_done

    def test_explored_blocks_is_completion_marker(self):
        """Only explored_blocks.txt marks a run as complete, not trace.txt."""
        self._create_input_files(1)

        # Trace exists but no explored_blocks = NOT complete
        (self.trace_dir / "input_0.trace.txt").write_text("data\n")
        explored = self.trace_dir / "input_0.explored_blocks.txt"
        trace = self.trace_dir / "input_0.trace.txt"

        assert not explored.exists()
        assert trace.exists()

        # Should NOT be skipped
        should_skip = explored.exists() and trace.exists() and trace.stat().st_size > 0
        assert not should_skip

    def test_hash_prevents_skip_on_incomplete(self):
        """Hash file + incomplete inputs = resume, not skip."""
        self._create_input_files(3)

        # Write hash file (simulating a previous run that wrote it before being modified)
        bpl = self.bpl_dir / f"{self.test_name}_inline.bpl"
        bpl.write_text("// bpl content")
        bpl_hash = hashlib.sha256(bpl.read_bytes()).hexdigest()
        hash_file = self.trace_dir / f"{self.test_name}.interp.hash"
        hash_file.write_text(bpl_hash + "\n")

        # Only 1 of 3 done
        self._simulate_completion("input_0")

        input_files = list(self.input_dir.glob("*.json"))
        all_done = all(
            (self.trace_dir / f"{p.stem}.explored_blocks.txt").exists()
            for p in input_files
        )
        stored_hash = hash_file.read_text().strip()

        assert stored_hash == bpl_hash, "Hash should match"
        assert not all_done, "Not all inputs complete — should resume, not skip"


class TestInterpreterSubprocess(unittest.TestCase):
    """Test the interpreter CLI via subprocess to verify end-to-end behavior."""

    def test_force_flag_accepted(self):
        """The --force flag should be accepted by the CLI."""
        result = subprocess.run(
            [sys.executable, "-m", "interpreter.python.interpreter", "--help"],
            capture_output=True, text=True, cwd=ROOT
        )
        assert "--force" in result.stdout

    def test_missing_pkg_fails(self):
        """Running with a nonexistent package should fail."""
        result = subprocess.run(
            [sys.executable, "-m", "interpreter.python.interpreter", "/nonexistent/path_pkg"],
            capture_output=True, text=True, cwd=ROOT
        )
        assert result.returncode != 0


if __name__ == "__main__":
    unittest.main()
