import pickle
import sys
from pathlib import Path

import pytest

# This repo IS the `interpreter` package. Add the parent of the repo root
# to sys.path so package imports work in local test runs.
REPO_ROOT = Path(__file__).resolve().parent.parent
PARENT = REPO_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
# Also add repo root for direct imports (parser, utils)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def pytest_addoption(parser):
    group = parser.getgroup("interpreter correctness")
    group.addoption("--run-slow", action="store_true", default=False,
                    help="run tests marked slow")
    group.addoption("--run-benchmark", action="store_true", default=False,
                    help="run tests marked benchmark/requires_compiled_package")
    group.addoption("--run-exhaustive", action="store_true", default=False,
                    help="run tests marked exhaustive")


def pytest_collection_modifyitems(config, items):
    gates = [
        ("slow", config.getoption("--run-slow"),
         "need --run-slow to run"),
        ("benchmark", config.getoption("--run-benchmark"),
         "need --run-benchmark to run"),
        ("requires_compiled_package", config.getoption("--run-benchmark"),
         "need --run-benchmark to run"),
        ("exhaustive", config.getoption("--run-exhaustive"),
         "need --run-exhaustive to run"),
    ]
    for item in items:
        for marker, enabled, reason in gates:
            if marker in item.keywords and not enabled:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def project_root():
    return PARENT


def available_benchmarks():
    """Return list of available benchmark names that have compiled packages."""
    pkg_dir = PARENT / "test_packages"
    if not pkg_dir.exists():
        return []
    benchmarks = []
    for pkg in sorted(pkg_dir.iterdir()):
        if pkg.is_dir() and pkg.name.endswith("_pkg"):
            name = pkg.name.removesuffix("_pkg")
            pkl = pkg / f"{name}.pkl"
            input_dir = PARENT / "test_input" / name
            if pkl.exists() and input_dir.exists() and list(input_dir.glob("*.input")):
                benchmarks.append(name)
    return benchmarks


@pytest.fixture(params=available_benchmarks())
def benchmark_name(request):
    return request.param


@pytest.fixture
def benchmark_data(benchmark_name):
    """Load program and first input for a benchmark."""
    pkg_dir = PARENT / "test_packages" / f"{benchmark_name}_pkg"
    pkl_path = pkg_dir / f"{benchmark_name}.pkl"
    input_dir = PARENT / "test_input" / benchmark_name
    input_file = sorted(input_dir.glob("*.input"))[0]

    with open(pkl_path, "rb") as f:
        program = pickle.load(f)

    from interpreter.utils.input_parser import get_bpl_field_sizes, parse_input_file

    field_sizes = get_bpl_field_sizes(pkg_dir, program=program)
    program_inputs = parse_input_file(input_file, field_sizes=field_sizes)

    return {
        "name": benchmark_name,
        "program": program,
        "program_inputs": program_inputs,
        "input_file": input_file,
        "input_name": input_file.stem,
    }
