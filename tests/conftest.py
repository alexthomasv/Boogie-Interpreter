"""Shared fixtures for interpreter tests."""
import pytest
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interpreter.python.interpreter import BoogieInterpreter, find_entry_point


@pytest.fixture
def project_root():
    return ROOT


def available_benchmarks():
    """Return list of available benchmark names that have compiled packages."""
    pkg_dir = ROOT / "test_packages"
    if not pkg_dir.exists():
        return []
    benchmarks = []
    for pkg in sorted(pkg_dir.iterdir()):
        if pkg.is_dir() and pkg.name.endswith("_pkg"):
            name = pkg.name.removesuffix("_pkg")
            pkl = pkg / f"{name}.pkl"
            input_dir = ROOT / "test_input" / name
            if pkl.exists() and input_dir.exists() and list(input_dir.glob("*.json")):
                benchmarks.append(name)
    return benchmarks


@pytest.fixture(params=available_benchmarks())
def benchmark_name(request):
    return request.param


@pytest.fixture
def benchmark_data(benchmark_name):
    """Load program and first input for a benchmark."""
    pkg_dir = ROOT / "test_packages" / f"{benchmark_name}_pkg"
    pkl_path = pkg_dir / f"{benchmark_name}.pkl"
    input_dir = ROOT / "test_input" / benchmark_name
    input_file = sorted(input_dir.glob("*.json"))[0]

    with open(pkl_path, 'rb') as f:
        program = pickle.load(f)

    from utils.utils import parse_inputs
    program_inputs = parse_inputs(input_file)

    return {
        'name': benchmark_name,
        'program': program,
        'program_inputs': program_inputs,
        'input_file': input_file,
        'input_name': input_file.stem,
    }
