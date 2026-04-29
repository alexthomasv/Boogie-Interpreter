"""In-process interpreter execution with coverage tracking.

Calls the BoogieInterpreter directly (same path as run_python in runner.py)
and returns the explored_blocks set without relying on environment attributes
that may not always be set.
"""

import signal
from contextlib import contextmanager

from interpreter.utils.inputs import ProgramInputs


@contextmanager
def _alarm(seconds: int):
    """SIGALRM-based timeout (Unix only)."""
    def _handler(signum, frame):
        raise TimeoutError(f"interpreter timeout after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class Evaluator:
    """Run the Boogie interpreter on a ProgramInputs and return block coverage."""

    def __init__(self, program, test_name: str, timeout: int = 30):
        self.program = program
        self.test_name = test_name
        self.timeout = timeout
        self._entry = None

    def _get_entry(self):
        if self._entry is None:
            from interpreter.runner import find_entry_point
            self._entry = find_entry_point(self.program)
        return self._entry

    def run(self, program_inputs: ProgramInputs, input_name: str = "_cov_eval") -> set:
        """Execute program_inputs and return the set of explored block names.

        Returns an empty set on timeout or any interpreter error.
        """
        try:
            with _alarm(self.timeout):
                from interpreter.python.Environment import Environment
                from interpreter.python.interpreter import BoogieInterpreter

                env = Environment(self.test_name, input_name)
                env.full_trace = False
                env.LOG_READ = False

                entry = self._get_entry()
                if entry is None:
                    return set()

                interp = BoogieInterpreter(env, program_inputs, input_name)
                interp.preprocess(self.program)
                interp.execute_procedure(entry)
                return set(interp.explored_blocks)
        except TimeoutError:
            return set()
        except Exception:
            return set()
