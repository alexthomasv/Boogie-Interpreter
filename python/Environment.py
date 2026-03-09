from interpreter.python.Context import Context
from interpreter.python.MemoryMap import MemoryMap
from interpreter.parser.declaration import StorageDeclaration, ConstantDeclaration
from interpreter.parser.expression import QuantifiedExpression, BinaryExpression
from interpreter.utils.utils import extract_boogie_variables
from collections import defaultdict
from pathlib import Path
import pickle
import os


class Environment:
    def __init__(self, test_name, input_name):
        self.input_name = input_name
        self.stackFrames = [Context()]
        self.alloc_addr = 0
        self.alloc_addr_shadow = 0

        # Traces all variables
        self.watch_all_vars = True

        # Text trace file (opened lazily if full_trace is enabled)
        self._trace_path = Path("positive_examples") / test_name / f"{input_name}.trace.txt"
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_file = None
        self.debug_print_trace = False
        self.curr_block = ""
        self.last_block = ""
        self.pc = 0
        self.LOG_READ = True
        self._trace_parts = []
        self._trace_size = 0

        # Columnar trace accumulator (non-shadow rows only)
        self._col_path = Path("positive_examples") / test_name / f"{input_name}.trace.pkl"
        self._col_vars = []
        self._col_values = []
        self._col_blocks = []
        self._col_last_blocks = []
        self._col_pcs = []
        self._col_op_types = []

        # Compact trace accumulator (pre-aggregated for fast Redis loading)
        self._compact_path = Path("positive_examples") / test_name / f"{input_name}.trace.compact.pkl"
        self._agg_pc_values = defaultdict(set)
        self._agg_block_values = defaultdict(set)
        self._agg_op_values = defaultdict(set)
        self._agg_pc_registry = defaultdict(set)
        self._agg_block_registry = defaultdict(set)
        self._agg_total = 0
        self.full_trace = False

        # Direct reference to the single frame's var_store for fast access
        self._var_store = self.stackFrames[0].var_store

    def close(self):
        if self.trace_file is not None:
            try:
                self.trace_file.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def push_frame(self):
        self.stackFrames.append(Context())
        self._var_store = self.stackFrames[-1].var_store

    def pop_frame(self):
        self.stackFrames.pop()
        self._var_store = self.stackFrames[-1].var_store

    def last_frame(self):
        return self.stackFrames[-1]

    def add_constraint(self, constraint):
        self.last_frame().add_constraint(constraint)

    def dump_buffer_trace(self, fsync=False):
        if self._trace_parts:
            if self.trace_file is None:
                self.trace_file = open(self._trace_path, "a")
            self.trace_file.write("".join(self._trace_parts))
            self._trace_parts = []
            self._trace_size = 0
        if fsync and self.trace_file is not None:
            print(f"flushing trace file")
            self.trace_file.flush()
            os.fsync(self.trace_file.fileno())

    def _append_trace(self, key, value, op_type):
        # Skip shadow variables — consumer (init_positive_examples) skips them
        if key[-7:] == '.shadow':
            return

        cb = self.curr_block or 'GLOBAL'

        # Text trace (backward compat, skippable via full_trace flag)
        if self.full_trace:
            lb = self.last_block or 'GLOBAL'
            line = f"{key} <- {hex(value)} {cb} {lb} {self.pc} {op_type}\n"
            self._trace_parts.append(line)
            self._trace_size += len(line)
            if self._trace_size > 50000:
                self.dump_buffer_trace(fsync=False)

        # Compact accumulation — raw values, pickled only at flush time
        # Uses tuple keys to avoid per-entry string formatting
        raw_val = (value, "")
        pc = self.pc
        self._agg_pc_values[(key, pc)].add(raw_val)
        self._agg_block_values[(key, cb)].add(raw_val)
        self._agg_op_values[(key, pc, op_type)].add(raw_val)
        self._agg_pc_registry[key].add(pc)
        self._agg_block_registry[key].add(cb)
        self._agg_total += 1

    def flush_columnar(self):
        """Write accumulated columnar trace data to a pickle file."""
        if not self._col_vars:
            return
        data = {
            'var': self._col_vars,
            'value': self._col_values,
            'block': self._col_blocks,
            'last_block': self._col_last_blocks,
            'pc': self._col_pcs,
            'op_type': self._col_op_types,
        }
        with open(self._col_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def flush_compact(self):
        """Write pre-aggregated compact trace to a pickle file.

        Pickling is deferred to here (not per-entry) for massive speedup.
        """
        if self._agg_total == 0:
            return
        # Convert tuple keys → string keys and pickle values at flush time
        pc_values = {
            f"positive_examples_{k}_{pc}": {pickle.dumps(v) for v in vals}
            for (k, pc), vals in self._agg_pc_values.items()
        }
        block_values = {
            f"positive_examples_{k}_{blk}": {pickle.dumps(v) for v in vals}
            for (k, blk), vals in self._agg_block_values.items()
        }
        op_values = {
            f"positive_examples_to_op_type_{k}_{pc}_{op}": {pickle.dumps(v) for v in vals}
            for (k, pc, op), vals in self._agg_op_values.items()
        }
        pc_registry = {
            f"positive_examples_to_pc_{k}": {pickle.dumps(p) for p in pcs}
            for k, pcs in self._agg_pc_registry.items()
        }
        block_registry = {
            f"positive_examples_to_block_{k}": {pickle.dumps(b) for b in blks}
            for k, blks in self._agg_block_registry.items()
        }
        data = {
            'pc_values': pc_values,
            'block_values': block_values,
            'op_values': op_values,
            'pc_registry': pc_registry,
            'block_registry': block_registry,
            'total': self._agg_total,
        }
        with open(self._compact_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Kept for backward compatibility
    def debug_print_set_value(self, key, value, op_type):
        if self.watch_all_vars:
            if self.debug_print_trace:
                print(f"{key} <- {value} {self.curr_block if self.curr_block else 'GLOBAL'} {self.last_block if self.last_block else 'GLOBAL'} {self.pc} {op_type}")
            self._append_trace(key, value, op_type)

    def set_concrete_value(self, key, value, silent):
        if key == "$CurrAddr":
            print(f"[detected alloc] $CurrAddr: {key} <- {value}")
            self.alloc_addr = value
        elif key == "$CurrAddr.shadow":
            print(f"[detected alloc] $CurrAddr.shadow: {key} <- {value}")
            self.alloc_addr_shadow = value
        if isinstance(value, str):
            print(f"[detected str] {key} <- {value}")
            value = int(value, 0)
        if not silent:
            self._append_trace(key, value, "W")
        self._var_store[key] = value

    def set_memory_map(self, key, value, silent):
        if key != value.name:
            value = value.copy(key)
        self._var_store[key] = value

    _set_dispatch = {
        int: set_concrete_value,
        bool: set_concrete_value,
        str: set_concrete_value,
        MemoryMap: set_memory_map,
    }

    def set_value(self, key, value, silent=False):
        handler = self._set_dispatch.get(type(value))
        assert handler is not None, f"Unknown type: {value} {type(value)}"
        handler(self, key, value, silent)

    def get_concrete_value(self, var_name):
        # Fast path: check top frame first (covers 99%+ of cases)
        val = self._var_store.get(var_name)
        if val is not None:
            return val
        # Slow path: search older frames
        for frame in reversed(self.stackFrames[:-1]):
            if var_name in frame.var_store:
                return frame.var_store[var_name]
        return None

    def handle_curr_addr(self, var_name, constraints):
        boogie_vars = set()
        for constraint in constraints:
            constraint_vars = extract_boogie_variables(constraint)
            boogie_vars.update(constraint_vars)

        print(f"boogie_vars: {boogie_vars}")

        if constraints:
            alloc_size_var = None
            for boogie_var in boogie_vars:
                if boogie_var.name.endswith("$n") or boogie_var.name.endswith("$n.shadow"):
                    alloc_size_var = boogie_var
                    break
            if alloc_size_var is None:
                print(f"No alloc size found for $CurrAddr {boogie_vars}")
                return 0
            alloc_size_val = self._var_store.get(alloc_size_var.name)

            if var_name == "$CurrAddr":
                alloc_ptr = self.alloc_addr
                self.alloc_addr = (alloc_ptr + alloc_size_val + 255) & ~255
                self._var_store[var_name] = self.alloc_addr
            elif var_name == "$CurrAddr.shadow":
                alloc_ptr = self.alloc_addr_shadow
                self.alloc_addr_shadow = (alloc_ptr + alloc_size_val + 255) & ~255
                self._var_store[var_name] = self.alloc_addr_shadow
        else:
            print(f"No constraints found for $CurrAddr {boogie_vars}")
        if var_name == "$CurrAddr":
            ret_val = self.alloc_addr
        else:
            ret_val = self.alloc_addr_shadow
        return ret_val

    def get_var(self, var_name, silent=False):
        # Fast path: check top frame directly (covers 99%+ of cases)
        val = self._var_store.get(var_name)
        if val is not None:
            if not isinstance(val, MemoryMap):
                if not silent and self.LOG_READ:
                    self._append_trace(var_name, val, "R")
            return val

        # Check older frames
        for frame in reversed(self.stackFrames[:-1]):
            if var_name in frame.var_store:
                val = frame.var_store[var_name]
                if not isinstance(val, MemoryMap):
                    if not silent and self.LOG_READ:
                        self._append_trace(var_name, val, "R")
                return val

        # $CurrAddr special handling
        if var_name == "$CurrAddr":
            curr_addr = self.alloc_addr
            if not silent and curr_addr and self.LOG_READ:
                self._append_trace(var_name, curr_addr, "R")
            return curr_addr
        if var_name == "$CurrAddr.shadow":
            curr_addr = self.alloc_addr_shadow
            if not silent and curr_addr and self.LOG_READ:
                self._append_trace(var_name, curr_addr, "R")
            return curr_addr

        print(f"No concrete value found for {var_name}, setting to 0")
        self.set_value(var_name, 0, silent)
        return 0

    def get_constraints(self, var_name) -> set:
        for frame in reversed(self.stackFrames):
            if var_name in frame.constraints:
                return frame.constraints[var_name]
        return set()

    def clear_var(self, var_name):
        for frame in reversed(self.stackFrames):
            if var_name in frame.var_store:
                if isinstance(frame.var_store[var_name], MemoryMap):
                    frame.var_store[var_name].clear()
                else:
                    del frame.var_store[var_name]
                return

    def dump_vars(self):
        for var, value in self._var_store.items():
            if not isinstance(value, MemoryMap):
                print(f"{var}: {value}\n")

    def dump_memory(self, file=None):
        for var, value in self._var_store.items():
            if isinstance(value, MemoryMap):
                output = f"{value}\n"
            else:
                output = f"{var}: {hex(value)}\n"

            if file:
                file.write(output)
            else:
                print(output, end="")
