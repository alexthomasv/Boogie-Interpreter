"""
Inspect the auth_length_pub benchmark by running the Boogie interpreter
and printing key output values.
"""

import sys
import os
import pickle
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.utils import parse_inputs
from interpreter.python.Environment import Environment
from interpreter.python.MemoryMap import MemoryMap
from interpreter.python.interpreter import BoogieInterpreter, find_entry_point


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Load the package
    pkg_path = os.path.join(base_dir, 'test_packages', 'auth_length_pub_pkg', 'auth_length_pub.pkl')
    with open(pkg_path, 'rb') as f:
        program = pickle.load(f)

    # 2. Load input
    input_path = os.path.join(base_dir, 'test_input', 'auth_length_pub', 'input_0.json')
    program_inputs = parse_inputs(input_path)

    # 3. Create interpreter with a temp trace file
    test_name = 'auth_length_pub'
    input_name = 'input_0'

    # Use a temp file for the trace (fsync requires a real file, not /dev/null)
    tmp_trace = tempfile.NamedTemporaryFile(mode='w', suffix='.trace.txt', delete=False)
    tmp_trace_path = tmp_trace.name

    # Create the Environment but override its trace file to use our temp file
    env = Environment(test_name, input_name)
    env.trace_file.close()
    env.trace_file = tmp_trace

    interp = BoogieInterpreter(env, program_inputs, input_name)
    interp.preprocess(program)

    entry = find_entry_point(program)
    assert entry is not None, "No entry point found"

    print(f"Executing entry procedure: {entry.name}")
    interp.execute_procedure(entry)

    # 4. Print results

    # Return values
    try:
        r_val = env.get_var('$r', silent=True)
        print(f"Return $r = {r_val} (hex: {hex(r_val) if isinstance(r_val, int) else r_val})")
    except Exception as e:
        print(f"Return $r = <not found: {e}>")

    try:
        r_shadow = env.get_var('$r.shadow', silent=True)
        print(f"Return $r.shadow = {r_shadow} (hex: {hex(r_shadow) if isinstance(r_shadow, int) else r_shadow})")
    except Exception as e:
        print(f"Return $r.shadow = <not found: {e}>")

    # Memory maps
    frame = env.last_frame()
    mem_maps = {}
    for name, val in frame.var_store.items():
        if isinstance(val, MemoryMap):
            mem_maps[name] = val

    print(f"\nMemory map names and entry counts:")
    for name in sorted(mem_maps.keys()):
        mm = mem_maps[name]
        print(f"  M${name}: {len(mm.memory)} entries")

    # Print first 32 entries for non-empty, non-shadow maps
    for name in sorted(mem_maps.keys()):
        if '.shadow' in name:
            continue
        mm = mem_maps[name]
        if len(mm.memory) == 0:
            continue
        print(f"\nM${name} (first 32 entries, addr -> value):")
        sorted_addrs = sorted(mm.memory.keys())[:32]
        for addr in sorted_addrs:
            val = mm.memory[addr]
            print(f"  M${name}[{hex(addr)}] = {hex(val)}")

    # Explored blocks
    print(f"\nExplored blocks: {len(interp.explored_blocks)}")

    # Cleanup
    interp.close()
    env.close()
    try:
        os.unlink(tmp_trace_path)
    except:
        pass


if __name__ == '__main__':
    main()
