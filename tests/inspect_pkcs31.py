#!/usr/bin/env python3
"""Inspect the bearssl_test_pkcs1_i31 benchmark by running the Boogie interpreter."""

import os
import sys
import pickle
import tempfile

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.utils import parse_inputs
from interpreter.python.Environment import Environment
from interpreter.python.MemoryMap import MemoryMap
from interpreter.python.interpreter import BoogieInterpreter, find_entry_point

# Paths
PKG_PATH = os.path.join(PROJECT_ROOT, 'test_packages', 'bearssl_test_pkcs1_i31_pkg', 'bearssl_test_pkcs1_i31.pkl')
INPUT_PATH = os.path.join(PROJECT_ROOT, 'test_input', 'bearssl_test_pkcs1_i31', 'test_01b_standard_2048.json')

TEST_NAME = 'bearssl_test_pkcs1_i31'
INPUT_NAME = 'test_01b_standard_2048'

print(f"Loading package from: {PKG_PATH}")
with open(PKG_PATH, 'rb') as f:
    program = pickle.load(f)

print(f"Loading inputs from: {INPUT_PATH}")
program_inputs = parse_inputs(INPUT_PATH)

print(f"Input variables: {list(program_inputs.variables.keys())}")
print()

# Create environment with a real temp file for trace
env = Environment(TEST_NAME, INPUT_NAME)

# Create interpreter
interp = BoogieInterpreter(env, program_inputs, TEST_NAME)

# Preprocess
print("=== PREPROCESSING ===")
interp.preprocess(program)

# Find entry point
entry = find_entry_point(program)
assert entry is not None, "No entry point found"
print(f"\nEntry point: {entry.name}")
print()

# Execute
print("=== EXECUTING ===")
interp.execute_procedure(entry)
print()

# Print results
print("=" * 60)
print("=== RESULTS ===")
print("=" * 60)

# Return values
print("\n--- Return Values ---")
for var in ['$r', '$r.shadow']:
    try:
        val = env.get_var(var, silent=True)
        print(f"  {var} = {val} (hex: {hex(val) if isinstance(val, int) else 'N/A'})")
    except Exception as e:
        print(f"  {var} = ERROR: {e}")

# Memory maps
print("\n--- Memory Maps ---")
frame = env.last_frame()
for name, val in sorted(frame.variables.items()):
    if isinstance(val, MemoryMap):
        print(f"  {name}: {len(val.memory)} entries (index_bw={val.index_bit_width}, elem_bw={val.element_bit_width})")

# Explored blocks
print(f"\n--- Explored Blocks ---")
print(f"  Number of explored blocks: {len(interp.explored_blocks)}")

# Specific variables
print("\n--- Key Variables ---")
for var in ['$p0', '$p0.shadow', '$p1', '$p1.shadow', '$p3', '$p3.shadow', '$i2', '$i2.shadow']:
    try:
        val = env.get_var(var, silent=True)
        if isinstance(val, int):
            print(f"  {var} = {val} (hex: {hex(val)})")
        elif isinstance(val, MemoryMap):
            print(f"  {var} = MemoryMap({val.name}, {len(val.memory)} entries)")
        else:
            print(f"  {var} = {val} (type: {type(val).__name__})")
    except Exception as e:
        print(f"  {var} = ERROR: {e}")

# Cleanup
interp.close()
env.close()

print("\nDone.")
