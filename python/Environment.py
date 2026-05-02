from interpreter.python.Context import Context
from interpreter.python.MemoryMap import MemoryMap
from interpreter.parser.declaration import StorageDeclaration, ConstantDeclaration
from interpreter.parser.expression import QuantifiedExpression, BinaryExpression
from interpreter.utils.utils import extract_boogie_variables
from interpreter.utils.raw_log import RawLogWriter
from interpreter.utils.debug_log import DebugLogger
from collections import defaultdict
from pathlib import Path
import os


class Environment:
    def __init__(self, test_name, input_name, debug_logger=None):
        self.input_name = input_name
        self.debug = (debug_logger or DebugLogger.disabled()).bind(
            input_name=input_name)
        self.stackFrames = [Context()]
        self.alloc_addr = 0
        self.alloc_addr_shadow = 0
        # Type-aware trace: store values matching declared types.
        # Built by interpreter from program declarations.
        self._var_types = {}       # var_name → Type object
        self._int_type_names = set()  # type names that alias to int (e.g. {"i32", "i8", ...})

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

        # Raw-log sink (the ONLY persistent trace output).  Must be
        # attached via ``enable_raw_log`` before any trace records are
        # generated — concretization writes happen during interpreter init
        # so the sink is installed immediately after construction.
        self._raw_log: RawLogWriter | None = None
        self._raw_log_path = Path("positive_examples") / test_name / f"{input_name}.trace.raw.zst"
        self._var_id = {}     # var name -> u32 id (assigned lazily)
        self._var_names_list = []
        self._block_id = {}   # block name -> u32 id
        self._block_names_list = []
        # Monotonic per-(var, block) write counter for loop-iteration ids.
        self._write_counter = defaultdict(int)  # (var, block) → count
        # Total trace events (reads + writes); exposed via _agg_total for
        # backwards-compatible callers.
        self._agg_total = 0
        self._raw_log_count = 0
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
            self.debug.event("trace", "text_trace_flush", path=self._trace_path,
                             bytes=self._trace_size, fsync=True)
            print(f"flushing trace file")
            self.trace_file.flush()
            os.fsync(self.trace_file.fileno())

    def enable_raw_log(self, var_names: list[str], block_names: list[str]):
        """Create the raw-log writer with a pre-built var + block table.

        The interpreter walks the program up front (``BoogieInterpreter``
        does this via ``generate_label_to_block`` + the list of
        declarations) and hands the complete name tables here so the
        streaming header can be written immediately — records are then
        appended as execution proceeds.

        Names that appear later for any reason (e.g. a read of an uninitialised
        var) are appended to the tables; the header already contains them.
        """
        if self._raw_log is not None:
            return
        self._raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        writer = RawLogWriter(self._raw_log_path)
        self._var_names_list = list(var_names)
        self._var_id = {n: i for i, n in enumerate(self._var_names_list)}
        self._block_names_list = list(block_names)
        self._block_id = {n: i for i, n in enumerate(self._block_names_list)}
        writer.write_header(self._var_names_list, self._block_names_list)
        self._raw_log = writer
        self.debug.event(
            "trace",
            "raw_log_open",
            path=self._raw_log_path,
            vars=len(self._var_names_list),
            blocks=len(self._block_names_list),
        )

    def _intern_var(self, name: str) -> int:
        vid = self._var_id.get(name)
        if vid is None:
            # A name we did not see during upfront scan. Shouldn't
            # normally happen — if it does, we can't change the header,
            # so the loader won't know about it.  Fall back to emitting
            # a zero id and an error at finish time.
            vid = len(self._var_names_list)
            self._var_id[name] = vid
            self._var_names_list.append(name)
            self._late_vars = True
            self.debug.event("trace", "raw_log_late_var", var=name, var_id=vid)
        return vid

    def _intern_block(self, name: str) -> int:
        bid = self._block_id.get(name)
        if bid is None:
            bid = len(self._block_names_list)
            self._block_id[name] = bid
            self._block_names_list.append(name)
            self._late_blocks = True
            self.debug.event("trace", "raw_log_late_block", block=name, block_id=bid)
        return bid

    def _append_trace(self, key, value, op_type):
        # Skip shadow variables — downstream consumers skip them anyway.
        if key[-7:] == '.shadow':
            return

        # Store value consistent with variable's declared type.
        # Integer-typed vars → signed. Bitvector-typed vars → unsigned.
        vtype = self._var_types.get(key)
        if vtype is not None:
            from interpreter.parser.type import IntegerType, CustomType
            if isinstance(vtype, IntegerType):
                if value >= (1 << 63):
                    value = value - (1 << 64)
            elif isinstance(vtype, CustomType):
                if vtype.name.startswith('bv'):
                    width = int(vtype.name[2:])
                    value = value & ((1 << width) - 1)
                elif vtype.name in self._int_type_names:
                    if value >= (1 << 63):
                        value = value - (1 << 64)

        cb = self.curr_block or 'GLOBAL'

        # Text trace (backward compat, skippable via full_trace flag)
        if self.full_trace:
            lb = self.last_block or 'GLOBAL'
            line = f"{key} <- {hex(value)} {cb} {lb} {self.pc} {op_type}\n"
            self._trace_parts.append(line)
            self._trace_size += len(line)
            if self._trace_size > 50000:
                self.dump_buffer_trace(fsync=False)

        # The Python interpreter is used for init-phase concretization
        # only — never inside a loop.  Emitting iter_id=0 matches the
        # Rust "not in any loop" convention, and downstream consumers
        # treat it as "no iteration info".
        iteration_id = 0

        self._agg_total += 1

        if self._raw_log is None:
            return

        kind = ord('W') if op_type == 'W' else ord('R')
        var_id = self._intern_var(key)
        block_id = self._intern_block(cb)
        self._raw_log.record(kind, var_id, self.pc, block_id, int(value), iteration_id)

    def flush_raw_log(self):
        """Close the raw-log writer.  Returns the record count."""
        if self._raw_log is None:
            return 0
        if getattr(self, '_late_vars', False) or getattr(self, '_late_blocks', False):
            self.debug.event(
                "trace",
                "raw_log_stale_header",
                late_vars=self._var_names_list[-10:],
                late_blocks=self._block_names_list[-10:],
            )
            raise RuntimeError(
                "raw-log: saw names not in the upfront scan — header is stale. "
                f"late_vars={self._var_names_list[-10:]!r} "
                f"late_blocks={self._block_names_list[-10:]!r}"
            )
        count = self._raw_log.finish()
        self._raw_log_count = count
        self._raw_log = None
        self.debug.event("trace", "raw_log_close", records=count,
                         path=self._raw_log_path)
        return count

    # Kept for backward compatibility
    def debug_print_set_value(self, key, value, op_type):
        if self.watch_all_vars:
            if self.debug_print_trace:
                print(f"{key} <- {value} {self.curr_block if self.curr_block else 'GLOBAL'} {self.last_block if self.last_block else 'GLOBAL'} {self.pc} {op_type}")
            self._append_trace(key, value, op_type)

    def set_concrete_value(self, key, value, silent):
        if key == "$CurrAddr":
            self.debug.event("mem", "alloc_addr_write", var=key, value=value)
            print(f"[detected alloc] $CurrAddr: {key} <- {value}")
            self.alloc_addr = value
        elif key == "$CurrAddr.shadow":
            self.debug.event("mem", "alloc_addr_write", var=key, value=value)
            print(f"[detected alloc] $CurrAddr.shadow: {key} <- {value}")
            self.alloc_addr_shadow = value
        if isinstance(value, str):
            self.debug.event("exec", "string_value_coerced", var=key, value=value)
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
        self.debug.event("exec", "uninitialized_var_defaulted",
                         var=var_name, value=0, pc=self.pc,
                         block=self.curr_block or "GLOBAL")
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
