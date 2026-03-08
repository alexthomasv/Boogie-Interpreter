#!/usr/bin/env python3
"""Inspect the BoogieInterpreter output for the bearssl_test_pkcs1_i15 benchmark."""

import os
import sys
import pickle
import tempfile

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from utils.utils import parse_inputs
from interpreter.python.Environment import Environment
from interpreter.python.interpreter import BoogieInterpreter, find_entry_point

# --- Configuration ---
PKG_PATH = os.path.join(PROJECT_ROOT, "test_packages", "bearssl_test_pkcs1_i15_pkg", "bearssl_test_pkcs1_i15.pkl")
INPUT_PATH = os.path.join(PROJECT_ROOT, "test_input", "bearssl_test_pkcs1_i15", "test_06_custom_vector.json")

TEST_NAME = "bearssl_test_pkcs1_i15"
INPUT_NAME = "test_06_custom_vector"

# --- Load program ---
print(f"Loading package: {PKG_PATH}")
with open(PKG_PATH, 'rb') as f:
    program = pickle.load(f)

# --- Load inputs ---
print(f"Loading inputs: {INPUT_PATH}")
program_inputs = parse_inputs(INPUT_PATH)

# --- Create Environment with a real temp trace file ---
# We monkey-patch Environment.__init__ to use a temp file instead of
# writing to positive_examples/
_orig_env_init = Environment.__init__

def _patched_env_init(self, test_name, input_name):
    from interpreter.python.Context import Context
    from pathlib import Path
    self.input_name = input_name
    self.stackFrames = [Context()]
    self.watch_vars = []
    self.alloc_addr = 0
    self.alloc_addr_shadow = 0
    self.watch_all_vars = False  # Disable tracing for speed
    self._tmp_trace = tempfile.NamedTemporaryFile(mode='w', suffix='.trace.txt', delete=False)
    self.trace_file = self._tmp_trace
    self.debug_print_trace = False
    self.curr_block = ""
    self.last_block = ""
    self.pc = 0
    self.LOG_READ = True
    self.buffer_trace = ""

Environment.__init__ = _patched_env_init

env = Environment(TEST_NAME, INPUT_NAME)
interp = BoogieInterpreter(env, program_inputs, INPUT_NAME)

print("=" * 60)
print("PREPROCESSING")
print("=" * 60)
interp.preprocess(program)

entry = find_entry_point(program)
assert entry is not None, "No entry point found"
print(f"\nEntry point: {entry.name}")

print("=" * 60)
print("EXECUTING")
print("=" * 60)
interp.execute_procedure(entry)

print("=" * 60)
print("RESULTS")
print("=" * 60)

# Return values
for var in ["$r", "$r.shadow"]:
    try:
        val = env.get_var(var, silent=True)
        print(f"  {var} = {val}")
    except Exception as e:
        print(f"  {var} = ERROR: {e}")

# Memory maps
print(f"\nMemory Maps:")
all_vars = env.last_frame().variables
mem_maps = {k: v for k, v in all_vars.items() if hasattr(v, 'entries')}
for name, mm in sorted(mem_maps.items()):
    print(f"  {name}: {len(mm.entries)} entries")

# Explored blocks
print(f"\nExplored blocks: {len(interp.explored_blocks)}")

# Key variables
print(f"\nKey Variables:")
for var in ["$p0", "$p0.shadow", "$p1", "$p1.shadow", "$p3", "$p3.shadow", "$i2", "$i2.shadow"]:
    try:
        val = env.get_var(var, silent=True)
        print(f"  {var} = {val}")
    except Exception as e:
        print(f"  {var} = ERROR: {e}")

# Clean up temp file
tmp_path = env._tmp_trace.name
env.close()
try:
    os.unlink(tmp_path)
except Exception:
    pass

print("\nDone.")
