"""
Inspect the bearssl_test_pms_ecdh benchmark by running the Boogie interpreter
and printing key output values.
"""
import os
import sys
import pickle
import tempfile

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from interpreter.python.Environment import Environment
from interpreter.python.MemoryMap import MemoryMap
from interpreter.python.interpreter import BoogieInterpreter, find_entry_point
from interpreter.utils.utils import parse_inputs


def main():
    test_name = "bearssl_test_pms_ecdh"
    pkg_path = os.path.join(PROJECT_ROOT, "test_packages", f"{test_name}_pkg", f"{test_name}.pkl")
    input_path = os.path.join(PROJECT_ROOT, "test_input", test_name, "input_0.json")

    # Load program
    print(f"Loading package from: {pkg_path}")
    with open(pkg_path, 'rb') as f:
        program = pickle.load(f)
    print("Package loaded successfully.")

    # Load inputs
    print(f"Loading inputs from: {input_path}")
    program_inputs = parse_inputs(input_path)
    print(f"Parsed {len(program_inputs.variables)} input variables: {list(program_inputs.variables.keys())}")

    # Create environment and interpreter
    # Use a temp directory for the input_name so trace files go to a temp location
    input_name = "input_0"
    environment = Environment(test_name, input_name)
    interp = BoogieInterpreter(environment, program_inputs, input_name)

    # Preprocess
    print("\n--- Preprocessing ---")
    interp.preprocess(program)

    # Find entry point
    entry = find_entry_point(program)
    assert entry is not None, "No entry point found"
    print(f"Entry point: {entry.name}")

    # Execute
    print("\n--- Executing ---")
    interp.execute_procedure(entry)
    print("\n--- Execution complete ---")

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Return values
    print("\n--- Return Values ---")
    for var in ["$r", "$r.shadow"]:
        val = environment.get_concrete_value(var)
        if val is not None:
            print(f"  {var} = {val} (hex: {hex(val) if isinstance(val, int) else val})")
        else:
            print(f"  {var} = None (not set)")

    # Memory maps
    print("\n--- Memory Maps ---")
    for frame in environment.stackFrames:
        for name, val in sorted(frame.var_store.items()):
            if isinstance(val, MemoryMap):
                print(f"  {name}: {len(val.memory)} entries (index_bw={val.index_bit_width}, elem_bw={val.element_bit_width})")

    # Number of explored blocks
    print(f"\n--- Explored Blocks ---")
    print(f"  Count: {len(interp.explored_blocks)}")

    # Key variable values
    print("\n--- Key Variables ---")
    for var in ["$p0", "$p0.shadow", "$i1", "$i1.shadow", "$i2", "$i2.shadow"]:
        val = environment.get_concrete_value(var)
        if val is not None:
            if isinstance(val, int):
                print(f"  {var} = {val} (hex: {hex(val)})")
            else:
                print(f"  {var} = {val}")
        else:
            print(f"  {var} = None (not set)")

    # Cleanup
    interp.close()
    environment.close()


if __name__ == "__main__":
    main()
