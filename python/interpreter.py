import argparse
import hashlib
import shutil
from interpreter.parser.expression import (
    FunctionApplication, MapSelect, StorageIdentifier, ProcedureIdentifier,
    BinaryExpression, UnaryExpression, BooleanLiteral, IntegerLiteral,
    OldExpression, LogicalNegation, QuantifiedExpression, IfExpression,
)
from interpreter.parser.statement import (
    AssertStatement, AssumeStatement, AssignStatement, Block,
    CallStatement, GotoStatement, HavocStatement, ReturnStatement,
)
from interpreter.parser.declaration import (
    StorageDeclaration, ImplementationDeclaration, ProcedureDeclaration,
    ConstantDeclaration, AxiomDeclaration,
)
from interpreter.parser.type import MapType
from collections import defaultdict
from interpreter.python.Environment import Environment
from interpreter.python.MemoryMap import MemoryMap
from interpreter.python.Buffer import ReadBuffer
from utils.utils import *
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import functools
import operator


# ============================================================
# Helpers
# ============================================================

def hex_to_bytes(s: str, *, as_bytearray: bool = True):
    s = "".join(s.split()).lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) % 2:
        raise ValueError("hex string must contain an even number of digits")
    blob = bytes.fromhex(s)
    return bytearray(blob) if as_bytearray else blob


def find_entry_point(program):
    for decl in program.declarations:
        if isinstance(decl, ImplementationDeclaration) and decl.has_attribute("entrypoint"):
            return decl
    return None


_STORE_FNS = frozenset(["$store.i8", "$store.i16", "$store.i32", "$store.i64", "$store.ref"])
_LOAD_FNS = frozenset(["$load.i8", "$load.i16", "$load.i32", "$load.i64", "$load.ref"])

# Pre-extract bit widths for store/load to avoid repeated string splitting
_STORE_BW = {fn: boogie_type_bitwidth[fn.split(".")[-1]] for fn in _STORE_FNS}
_LOAD_BW = {fn: boogie_type_bitwidth[fn.split(".")[-1]] for fn in _LOAD_FNS}

# Mask for 64-bit operations
_MASK_64 = (1 << 64) - 1


# ============================================================
# BoogieInterpreter
# ============================================================

class BoogieInterpreter:
    def __init__(self, environment: Environment, program_inputs, test_name: str):
        self.test_name = test_name
        self.env = environment
        # Accept either ProgramInputs or legacy dict[str, Input]
        if isinstance(program_inputs, ProgramInputs):
            self.inputs = program_inputs
            self.program_inputs = program_inputs.with_shadows()
        else:
            self.inputs = None
            self.program_inputs = program_inputs
        self.label_to_block = {}
        self.arr_inputs = []
        self.explored_blocks = set()
        extra = self.inputs.extra_data if self.inputs else program_inputs.get("extra_data")
        if extra:
            self.external_buffer = ReadBuffer(extra)
        else:
            self.external_buffer = ReadBuffer()
        self.fn_to_op = generate_function_map()

        self.wlen_buf = [209, 42, 6, 37, 51, 23]
        self.wlen_buf_idx = 0

        trace_dir = Path("mem_ops_traces") / test_name
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.memset_trace_file = (trace_dir / "memset_trace.txt").open("a")
        self.memset_trace = ""
        self.memcpy_trace_file = (trace_dir / "memcpy_trace.txt").open("a")
        self.memcpy_trace = ""
        self.read_trace_file = (trace_dir / "read_trace.txt").open("a")
        self.read_trace = ""

        # Cache frequently-used references to avoid attribute lookups in hot loop
        self._get_var = environment.get_var
        self._set_value = environment.set_value
        self._fn_to_op = self.fn_to_op

    def close(self):
        for f in (self.memset_trace_file, self.memcpy_trace_file, self.read_trace_file):
            try:
                f.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    # ----------------------------------------------------------
    # Preprocessing & initialization
    # ----------------------------------------------------------

    def preprocess(self, program):
        impl_decls = [d for d in program.declarations if isinstance(d, ImplementationDeclaration) and d.body]
        impl_names = {d.name for d in impl_decls}
        proc_decls = [d for d in program.declarations
                      if isinstance(d, ProcedureDeclaration)
                      and not isinstance(d, ImplementationDeclaration)
                      and d.name in impl_names]
        assert len(proc_decls) == 1 and len(impl_decls) == 1, (
            f"We only support inlined procedures. "
            f"{[p.name for p in proc_decls]} {len(proc_decls)} {len(impl_decls)} "
            f"{[d.name for d in program.declarations if isinstance(d, ProcedureDeclaration)]}"
        )

        self.impl_decl = impl_decls[0]
        self.impl_decl_name = impl_decls[0].name
        self.proc_decl = proc_decls[0]

        self.label_to_block = generate_label_to_block(program)
        self.pc_to_stmt, self.label_to_pc, self.pc_to_block_name, self.pc_to_label = (
            initialize_code_metadata(self.impl_decl)
        )
        self._initialize_global_axioms(program)
        self._initialize_vars(program.declarations, kind="global")
        self._concretize_global_memory()
        self.arr_inputs, self.field_inputs = preprocess_external_inputs(self.impl_decl)

    def _initialize_global_axioms(self, program):
        for decl in program.declarations:
            if isinstance(decl, AxiomDeclaration):
                self.env.add_constraint(decl.expression)

    def _initialize_vars(self, declarations, kind="global"):
        """Initialize variables from declarations (works for both global and local)."""
        for decl in declarations:
            if isinstance(decl, StorageDeclaration):
                if isinstance(decl.type, MapType):
                    assert len(decl.names) == 1
                    if kind == "global":
                        print(f"Initializing global memory: {decl.names[0]} {type(decl.names[0])}")
                    bw = convert_type_to_bitwidth(decl.type)
                    assert len(bw) == 2, f"Expected 2 elements for map type: {decl.type}"
                    self.env.set_value(
                        decl.names[0],
                        MemoryMap(self.test_name, decl.names[0], bw[0], bw[1], self.env),
                    )
                else:
                    self.env.set_value(decl.names[0], 0, silent=True)
            elif isinstance(decl, ConstantDeclaration):
                if kind == "global":
                    print(f"Initializing global constant: {decl.names[0]} {type(decl.names[0])}")
                self.env.set_value(decl.names[0], 0, silent=True)
            elif kind == "local":
                assert False, f"Unknown declaration type: {decl}"

    def _concretize_global_memory(self):
        constraints = self.env.last_frame().constraints
        for var in constraints:
            for constraint in constraints[var]:
                if isinstance(constraint, BinaryExpression) and constraint.op == "==":
                    lhs = constraint.lhs
                    rhs = self.eval(constraint.rhs)
                    if isinstance(lhs, StorageIdentifier) or isinstance(rhs, StorageIdentifier):
                        print(f"Concretizing global memory for {lhs.name} {rhs}")
                        self.env.set_value(lhs.name, rhs)

    def concretize_proc_inputs(self, procedure):
        for param in procedure.parameters:
            assert len(param.names) == 1
            name = param.names[0]
            if name in self.program_inputs and self.program_inputs[name].value is not None:
                print(f"Concretizing procedure input: {name} to {self.program_inputs[name]}")
                self.env.set_value(name, self.program_inputs[name].value)
            else:
                print(f"No value for {name}")
                self.env.set_value(name, 0, silent=True)

    # ----------------------------------------------------------
    # Memory concretization
    # ----------------------------------------------------------

    def _allocate_addresses(self, alloc_requests):
        """Assign concrete addresses to array base pointers, ensuring disjointness.

        Args:
            alloc_requests: list of (ArrayInfo, buffer_size_bytes) tuples
        """
        # Group by memory map, track max size per base_ptr
        maps = defaultdict(lambda: defaultdict(int))
        for arr_info, buf_size in alloc_requests:
            current = maps[arr_info.mem_map][arr_info.base_ptr]
            if buf_size > current:
                maps[arr_info.mem_map][arr_info.base_ptr] = buf_size

        ptr_assignments = {}
        print(f"{'MEM MAP':<15} | {'POINTER':<15} | {'ADDRESS':<10} | {'SIZE':<10}")
        print("-" * 60)

        for mem_map, ptr_dict in maps.items():
            current_addr = 0
            for ptr in sorted(ptr_dict.keys()):
                size = ptr_dict[ptr]
                ptr_assignments[ptr] = current_addr
                print(f"{mem_map:<15} | {ptr:<15} | {hex(current_addr):<10} | {size:<10}")
                current_addr += (size + 7) & ~7

        return ptr_assignments

    def concretize_input_memory(self):
        """Concretize all input buffers and struct fields into memory maps.

        Handles three kinds of Input:
          - buffers: simple array data (e.g. OID, hash, output buffer)
          - struct:  ordered struct fields with scalars and pointer-to-buffer fields
          - (scalars are handled by concretize_proc_inputs)
        """
        print("[concretize_input_memory] Concretizing input memory")

        # Phase 1: Collect allocation requests with correct sizes from JSON
        alloc_requests = []  # list of (ArrayInfo, buffer_size_bytes)

        for var_name, inp in self.program_inputs.items():
            if '.shadow' in var_name:
                continue
            arr_key = var_name
            shadow_key = f"{var_name}.shadow"

            if inp.buffers and arr_key in self.arr_inputs:
                non_shadow = [a for a in self.arr_inputs[arr_key] if '.shadow' not in a.mem_map]
                shadow = [a for a in self.arr_inputs.get(shadow_key, []) if '.shadow' in a.mem_map]
                for buf, ai in zip(inp.buffers, non_shadow):
                    alloc_requests.append((ai, buf['size']))
                for buf, ai in zip(inp.buffers, shadow):
                    alloc_requests.append((ai, buf['size']))

            if inp.struct and arr_key in self.arr_inputs:
                arr_non_shadow = [a for a in self.arr_inputs[arr_key] if '.shadow' not in a.mem_map]
                arr_shadow = [a for a in self.arr_inputs.get(shadow_key, []) if '.shadow' in a.mem_map]
                buf_idx = 0
                for field in inp.struct:
                    if 'buffer' in field:
                        buf_size = field['buffer']['size']
                        if buf_idx < len(arr_non_shadow):
                            alloc_requests.append((arr_non_shadow[buf_idx], buf_size))
                        if buf_idx < len(arr_shadow):
                            alloc_requests.append((arr_shadow[buf_idx], buf_size))
                        buf_idx += 1

        # Phase 2: Allocate and set addresses
        ptr_to_vals = self._allocate_addresses(alloc_requests)
        for ptr, val in ptr_to_vals.items():
            self.env.set_value(ptr, val)

        # Phase 3: Write data
        for var_name, inp in self.program_inputs.items():
            if '.shadow' in var_name:
                continue
            if inp.buffers:
                self._concretize_buffers(var_name, inp)
            if inp.struct:
                self._concretize_struct(var_name, inp)

    def _concretize_buffers(self, var_name, inp):
        """Write simple buffer data (non-struct arrays) to memory maps."""
        if var_name not in self.arr_inputs:
            print(f"[WARNING] {var_name} has buffer data but no BPL {{:array}} annotations")
            return

        non_shadow = [a for a in self.arr_inputs[var_name] if '.shadow' not in a.mem_map]
        shadow = [a for a in self.arr_inputs.get(f"{var_name}.shadow", [])
                  if '.shadow' in a.mem_map]

        for datum, arr_info in zip(inp.buffers, non_shadow):
            self._write_buffer(datum, arr_info)
        for datum, arr_info in zip(inp.buffers, shadow):
            self._write_buffer(datum, arr_info)

    def _concretize_struct(self, var_name, inp):
        """Write struct fields and their pointed-to buffers to memory maps."""
        # Get BPL field annotations (keyed by {:name} = var_name)
        field_infos = self.field_inputs.get(var_name, [])
        field_infos_shadow = self.field_inputs.get(f"{var_name}.shadow", [])

        # Get BPL array annotations for buffers pointed to by this struct
        arr_infos = self.arr_inputs.get(var_name, [])
        arr_infos_shadow = self.arr_inputs.get(f"{var_name}.shadow", [])
        # Non-shadow arrays only (for matching with JSON buffers)
        arr_non_shadow = [a for a in arr_infos if '.shadow' not in a.mem_map]
        arr_shadow = [a for a in arr_infos_shadow if '.shadow' in a.mem_map]

        # All struct fields (scalar + pointer) should match BPL field annotations
        assert len(inp.struct) == len(field_infos), (
            f"Struct field count mismatch for {var_name}: "
            f"JSON has {len(inp.struct)} fields, BPL has {len(field_infos)} {{:field}} annotations"
        )

        # Write scalar fields (and shadows)
        field_idx = 0
        buffer_idx = 0
        for struct_field in inp.struct:
            if 'value' in struct_field:
                # Scalar field — write value to memory map
                self._write_field(struct_field, field_infos[field_idx])
                if field_infos_shadow:
                    self._write_field(struct_field, field_infos_shadow[field_idx])
                field_idx += 1
            elif 'buffer' in struct_field:
                # Pointer field — value is the allocated address of the buffer
                arr_info = arr_non_shadow[buffer_idx]
                addr = self.eval(arr_info.offset)
                # Create a synthetic scalar field with the address as value
                addr_hex = f"0x{addr:0{struct_field['size'] * 2}x}"
                addr_field = {'value': addr_hex, 'size': struct_field['size']}
                self._write_field(addr_field, field_infos[field_idx])
                if field_infos_shadow:
                    self._write_field(addr_field, field_infos_shadow[field_idx])
                field_idx += 1

                # Write the buffer contents to the memory map
                self._write_buffer(struct_field['buffer'], arr_info)
                if buffer_idx < len(arr_shadow):
                    self._write_buffer(struct_field['buffer'], arr_shadow[buffer_idx])
                buffer_idx += 1

    def _write_buffer(self, datum, arr_info):
        """Write buffer contents from JSON into a memory map via ArrayInfo."""
        contents = hex_to_bytes(datum['contents'])
        mem_map = self.env.get_var(arr_info.mem_map, silent=True)
        base = self.eval(arr_info.offset)
        size = datum['size']

        for i in range(size):
            addr = base + i
            val = contents[i] if i < len(contents) else 0
            print(f"{mem_map.name}[{addr}] <- {hex(val)}")
            mem_map.set(addr, val)

    def _write_field(self, field_data, field_info):
        """Write a scalar field value into a memory map via FieldInfo."""
        contents = hex_to_bytes(field_data['value'])[::-1]  # little-endian
        mem_map = self.env.get_var(field_info.mem_map, silent=True)
        element_bit_width = mem_map.element_bit_width
        elt_bytes = element_bit_width // 8

        assert len(contents) == field_info.size, (
            f"Field size mismatch: data={len(contents)}, annotation={field_info.size}"
        )

        chunks = [int.from_bytes(contents[i:i + elt_bytes], byteorder="little")
                  for i in range(0, len(contents), elt_bytes)]

        base_ptr = self.eval(field_info.base_ptr)
        for i, chunk in enumerate(chunks):
            addr = base_ptr + i * element_bit_width
            print(f"Concretizing field: {field_info.mem_map}[{addr}] = {hex(chunk)}")
            mem_map.set(addr, chunk)

    # ----------------------------------------------------------
    # Execution engine
    # ----------------------------------------------------------

    def execute_procedure(self, procedure):
        self._initialize_vars(procedure.body.locals, kind="local")
        self.concretize_proc_inputs(procedure)
        self.concretize_input_memory()
        print(f"[execute_procedure] Concretized input memory")

        # Cache for the hot loop
        label_to_block = self.label_to_block
        label_to_pc = self.label_to_pc
        explored_blocks = self.explored_blocks
        env = self.env
        execute_block = self._execute_block

        blocks = [procedure.body.blocks[0]]
        while blocks:
            block = blocks.pop()
            explored_blocks.add(block.name)
            last_stmt = execute_block(block, env, label_to_pc)
            if type(last_stmt) is GotoStatement:
                targets = last_stmt.identifiers
                if len(targets) == 1:
                    blocks.append(label_to_block[targets[0].name])
                else:
                    self._handle_goto_branch(last_stmt, blocks)
            elif type(last_stmt) is ReturnStatement:
                break
            else:
                assert False, f"Unknown statement type: {type(last_stmt)}"

        env.dump_buffer_trace(fsync=True)
        env.flush_columnar()
        env.flush_compact()
        self._dump_mem_trace()

    def _execute_block(self, block, env, label_to_pc):
        env.last_block = env.curr_block
        env.curr_block = block.name
        env.pc = label_to_pc[block.name]
        stmts = block.statements
        n = len(stmts) - 1
        execute = self._execute_statement
        for i in range(n):
            execute(stmts[i])
            env.pc += 1
        return stmts[n]

    def _handle_goto_branch(self, stmt, blocks):
        targets = stmt.identifiers
        target_blocks = [self.label_to_block[t.name] for t in targets]
        taken = None
        for tblock in target_blocks:
            cond = extract_first_assume_stmt(tblock)
            assert isinstance(cond, AssumeStatement), f"{cond} {type(cond)}"
            if self.eval(cond.expression):
                assert taken is None, f"Multiple goto conditions are true: {stmt}"
                taken = tblock
        assert taken is not None, f"No goto condition is true: {stmt}"
        blocks.append(taken)

    # ----------------------------------------------------------
    # Statement dispatch
    # ----------------------------------------------------------

    def _handle_assign(self, stmt):
        # Fast path: single assignment (most common case)
        lhs_list = stmt.lhs
        rhs_list = stmt.rhs
        if len(lhs_list) == 1:
            self._set_value(lhs_list[0].name, self.eval(rhs_list[0]))
        else:
            values = [self.eval(r) for r in rhs_list]
            for l, v in zip(lhs_list, values):
                self._set_value(l.name, v)

    def _handle_call(self, stmt):
        proc = stmt.procedure.name
        if RE_PRINTF.match(proc):
            return
        if any(pat.match(proc) for pat in CALL_IGNORE_FN_PATTERNS):
            return
        if proc == "putc.cross_product":
            return
        if proc == "time.cross_product":
            [self.eval(arg) for arg in stmt.arguments]
            time_res = int(time.time())
            for v in stmt.assignments:
                self._set_value(v.name, time_res)
            return
        if proc == "write.cross_product":
            [self.eval(arg) for arg in stmt.arguments]
            for v in stmt.assignments:
                self._set_value(v.name, self.wlen_buf[self.wlen_buf_idx % len(self.wlen_buf)])
            self.wlen_buf_idx += 1
            return
        if proc == "read.cross_product":
            args = [self.eval(arg) for arg in stmt.arguments]
            buf = self._get_var("$M.0")
            buf_shadow = self._get_var("$M.0.shadow")
            buf_ptr, buf_ptr_shadow = args[2], args[3]
            read_len, read_len_shadow = args[4], args[5]
            assert read_len == read_len_shadow, f"TODO: Handle read.cross_product: {read_len} {read_len_shadow}"
            data = self.external_buffer.read(read_len)
            for i in range(read_len):
                buf.set(buf_ptr + i, data[i])
                buf_shadow.set(buf_ptr_shadow + i, data[i])
                self.read_trace += f"{buf.name}[{hex(buf_ptr + i)}] = {hex(data[i])}\n"
            if len(self.read_trace) > 100:
                self.read_trace_file.write(self.read_trace)
                self.read_trace = ""
            for v in stmt.assignments:
                self._set_value(v.name, read_len)
            return
        if proc == "crypto_memcmp":
            assert False, f"TODO: Handle crypto_memcmp: {stmt}"
        assert False, f"Unknown call statement: {proc}"

    def _handle_goto(self, stmt):
        assert False, f"Goto statement should be handled by caller: {stmt}"

    def _handle_assert(self, stmt):
        if not self.eval(stmt.expression):
            self.eval(stmt.expression, debug=True)
            print(f"Assertion {stmt.expression} is violated")
            self.env.dump_memory()
            assert False, f"Assert failed: {stmt}"

    def _handle_assume(self, stmt):
        expr = stmt.expression
        if type(expr) is BooleanLiteral and expr.value:
            return
        if type(expr) is QuantifiedExpression:
            self._handle_quantified_assume(expr)
        else:
            self.eval(expr)

    def _handle_havoc(self, stmt):
        names = {ident.name for ident in stmt.identifiers}
        if {"$CurrAddr", "$CurrAddr.shadow"} & names:
            assert len(names) == 1, f"Havoc on $CurrAddr must only contain one variable"
            block = self.label_to_block[self.env.curr_block]
            offset_pc = self.env.pc - self.label_to_pc[self.env.curr_block]
            havoc_var = stmt.identifiers[0].name
            # Scan forward for AssumeStatements that constrain this havoc var
            constraints = set()
            for j in range(offset_pc + 1, len(block.statements)):
                s = block.statements[j]
                if isinstance(s, AssumeStatement) and havoc_var in str(s.expression):
                    constraints.add(s.expression)
                elif not isinstance(s, (HavocStatement, AssumeStatement)):
                    break
            self.env.handle_curr_addr(havoc_var, constraints)
            self._get_var(havoc_var)

        for ident in stmt.identifiers:
            self.env.clear_var(ident.name)

    _stmt_dispatch = {
        AssignStatement: _handle_assign,
        CallStatement: _handle_call,
        HavocStatement: _handle_havoc,
        GotoStatement: _handle_goto,
        AssertStatement: _handle_assert,
        AssumeStatement: _handle_assume,
    }

    def _execute_statement(self, stmt):
        handler = self._stmt_dispatch.get(type(stmt))
        assert handler is not None, f"Unknown statement type: {stmt} ({type(stmt)})"
        handler(self, stmt)

    # Public alias for external callers (tests, etc.)
    execute_statement = _execute_statement

    # ----------------------------------------------------------
    # Quantified assume (memset / memcpy)
    # ----------------------------------------------------------

    def _handle_quantified_assume(self, q_expr):
        """Dispatch forall assume patterns for memset and memcpy."""
        boogie_vars = extract_boogie_variables(q_expr.expression)

        if any("memset" in v.name for v in boogie_vars):
            self._handle_memset_assume(q_expr, boogie_vars)
        elif any("memcpy" in v.name for v in boogie_vars):
            self._handle_memcpy_assume(q_expr, boogie_vars)
        else:
            assert False, f"TODO: Handle quantified expression: {q_expr.expression.op}"

    @staticmethod
    def _classify_vars(boogie_vars, suffix_map):
        result = {}
        for var in boogie_vars:
            for suffix, role in suffix_map.items():
                if var.name.endswith(suffix) or var.name.endswith(suffix + ".shadow"):
                    result[role] = var
                    break
        return result

    def _handle_memset_assume(self, q_expr, boogie_vars):
        expr = q_expr.expression

        if expr.op == "&&":
            free_vars = [v for v in boogie_vars if v not in q_expr.variables]
            classified = self._classify_vars(free_vars, {
                "M.ret": "M_ret", "dst": "dst", "len": "len", "val": "val",
            })
            for role in ("M_ret", "dst", "len", "val"):
                assert role in classified, f"Missing {role} in memset &&: {q_expr.expression}"
            assert "memset" in classified["dst"].name, f"Not memset?? {q_expr.expression}"

            val = self._get_var(classified["val"].name)
            dst = self._get_var(classified["dst"].name)
            range_len = self._get_var(classified["len"].name)
            M_ret = self._get_var(classified["M_ret"].name)
            for addr in range(dst, dst + range_len):
                M_ret.set(addr, val)

        elif expr.op == "==>":
            self._handle_preserve_assume(
                expr, boogie_vars, "memset",
                src_suffix="M", dst_suffix="M.ret",
            )
        else:
            assert False, f"TODO: Handle memset quantified expression: {expr.op}"

    def _handle_memcpy_assume(self, q_expr, boogie_vars):
        expr = q_expr.expression

        if expr.op == "&&":
            classified = self._classify_vars(boogie_vars, {
                "$M.ret": "M_ret", "$M.src": "M_src",
                "$len": "len", "$src": "src", "$dst": "dst",
            })
            for role in ("M_ret", "M_src", "len", "src", "dst"):
                assert role in classified, f"Missing {role} in memcpy &&: {boogie_vars}"

            dst = self._get_var(classified["dst"].name)
            src = self._get_var(classified["src"].name)
            range_len = self._get_var(classified["len"].name)
            M_ret = self._get_var(classified["M_ret"].name)
            M_src = self._get_var(classified["M_src"].name)
            for offset in range(range_len):
                M_ret.set(dst + offset, M_src.get(src + offset))

        elif expr.op == "==>":
            self._handle_preserve_assume(
                expr, boogie_vars, "memcpy",
                src_suffix="$M.dst", dst_suffix="$M.ret",
            )
        else:
            assert False, f"TODO: Handle memcpy quantified expression: {expr.op}"

    def _handle_preserve_assume(self, expr, boogie_vars, op_name, src_suffix, dst_suffix):
        lhs = expr.lhs
        fn_name = lhs.function.name

        if fn_name.startswith("$slt"):
            classified = self._classify_vars(boogie_vars, {
                dst_suffix: "M_ret", src_suffix: "M_src", "dst": "dst",
            })
            for role in ("M_ret", "M_src", "dst"):
                assert role in classified, f"Missing {role} in {op_name} slt: {boogie_vars}"
            if op_name == "memset":
                assert "memset" in classified["dst"].name, f"Not memset?? {expr}"

            dst = self._get_var(classified["dst"].name)
            M_ret = self._get_var(classified["M_ret"].name)
            M_src = self._get_var(classified["M_src"].name)
            for addr in M_src.memory:
                if addr < dst:
                    M_ret.set(addr, M_src.get(addr))

        elif fn_name.startswith("$sle"):
            suffix_map = {
                dst_suffix: "M_ret", src_suffix: "M_src",
                "dst": "dst", "len": "len",
            }
            classified = self._classify_vars(boogie_vars, suffix_map)
            for role in ("M_ret", "M_src", "dst", "len"):
                assert role in classified, f"Missing {role} in {op_name} sle: {boogie_vars}"
            if op_name == "memset":
                assert "memset" in classified["dst"].name, f"Not memset?? {expr}"

            dst = self._get_var(classified["dst"].name)
            range_len = self._get_var(classified["len"].name)
            M_ret = self._get_var(classified["M_ret"].name)
            M_src = self._get_var(classified["M_src"].name)
            boundary = dst + range_len
            for addr in M_src.memory:
                if boundary <= addr:
                    M_ret.set(addr, M_src.get(addr))
        else:
            assert False, f"TODO: Handle quantified expression: {expr}"

    # ----------------------------------------------------------
    # Expression evaluation (direct recursive — fast)
    # ----------------------------------------------------------

    # Binary op table: op_string -> (function, output_width)
    # output_width is bool for boolean ops, or int bit-width for arithmetic
    _binary_op_map = {
        "==": (operator.eq, bool),
        "!=": (operator.ne, bool),
        "<": (operator.lt, bool),
        ">": (operator.gt, bool),
        ">=": (operator.ge, bool),
        "<=": (operator.le, bool),
        "&&": (lambda x, y: x and y, bool),
        "||": (lambda x, y: x or y, bool),
        "==>": (lambda x, y: (not x) or y, bool),
        "<==>": (lambda x, y: x == y, bool),
        "-": (operator.sub, 64),
        "*": (operator.mul, 64),
        "+": (operator.add, 64),
    }

    def eval(self, expr, debug=False):
        """Evaluate an expression. Direct recursive for speed."""
        return self._eval(expr)

    def _eval(self, expr):
        """Core recursive evaluator — no stack, no id(), no dicts."""
        t = type(expr)

        # Leaf nodes (most common — ~60% of all nodes)
        if t is StorageIdentifier or t is ProcedureIdentifier:
            return self._get_var(expr.name)
        if t is IntegerLiteral or t is BooleanLiteral:
            return expr.value

        # Function applications (~25% of nodes)
        if t is FunctionApplication:
            f_name = expr.function.name
            fn_entry = self._fn_to_op.get(f_name)
            if fn_entry is not None:
                op, num_args, op_bw, out_bw = fn_entry
                if num_args == 1:
                    return op(self._eval(expr.arguments[0]))
                a = self._eval(expr.arguments[0])
                b = self._eval(expr.arguments[1])
                result = op(a, b)
                return bool(result) if out_bw is bool else result & _MASK_64

            if f_name in _STORE_FNS:
                return self._eval_store_fast(f_name, expr.arguments)
            if f_name in _LOAD_FNS:
                return self._eval_load_fast(f_name, expr.arguments)
            if f_name == "$isExternal":
                return 0
            if f_name == "read":
                self._eval_read(expr)
            elif f_name == "write":
                self._eval_write(expr)
            assert False, f"Unknown function application: {expr} {f_name}"

        # Binary expressions (~10% of nodes)
        if t is BinaryExpression:
            lhs = self._eval(expr.lhs)
            rhs = self._eval(expr.rhs)
            entry = self._binary_op_map.get(expr.op)
            assert entry is not None, f"[eval] Unsupported binary op: {expr.op}"
            op_func, out_width = entry
            result = op_func(lhs, rhs)
            return bool(result) if out_width is bool else result & _MASK_64

        # Negation
        if t is LogicalNegation:
            return not self._eval(expr.expression)

        # If-then-else
        if t is IfExpression:
            if self._eval(expr.condition):
                return self._eval(expr.then)
            return self._eval(expr.else_)

        # Unary (minus, etc.)
        if t is UnaryExpression:
            return self._eval(expr.expression)

        if t is OldExpression:
            assert False, f"TODO: Support old expression: {expr}"

        assert False, f"Unknown expression type: {expr} {t}"

    def _eval_store_fast(self, f_name, args):
        bit_width = _STORE_BW[f_name]
        memory_map = self._eval(args[0])
        index = self._eval(args[1])
        value = self._eval(args[2])
        ew = memory_map.element_bit_width
        ew_mask = memory_map._element_mask

        if bit_width == ew:
            memory_map.set(index, value)
        else:
            for i in range(bit_width // ew):
                memory_map.set(index + i, (value >> (i * ew)) & ew_mask)
        return memory_map

    def _eval_load_fast(self, f_name, args):
        bit_width = _LOAD_BW[f_name]
        memory_map = self._eval(args[0])
        index = self._eval(args[1])
        ew = memory_map.element_bit_width

        if bit_width == ew:
            return memory_map.get(index)

        result = 0
        for i in range(bit_width // ew):
            result |= memory_map.get(index + i) << (i * ew)
        return result

    def _eval_read(self, expr):
        fd = self._eval(expr.arguments[0])
        memory_map = self._eval(expr.arguments[1])
        read_len = self._eval(expr.arguments[2])
        data = self.external_buffer.read(read_len)
        print(f"Reading {read_len} bytes from {data}")
        for i in range(read_len):
            memory_map.set(i, data[i])
        assert False, f"TODO: Handle read: {expr}"

    def _eval_write(self, expr):
        fd = self._eval(expr.arguments[0])
        memory_map = self._eval(expr.arguments[1])
        write_len = self._eval(expr.arguments[2])
        print(f"Writing {write_len} bytes to {memory_map}")
        assert False, f"TODO: Handle write: {expr}"

    # ----------------------------------------------------------
    # Trace dumping
    # ----------------------------------------------------------

    def _dump_mem_trace(self):
        self.memset_trace_file.write(self.memset_trace)
        self.memcpy_trace_file.write(self.memcpy_trace)
        self.read_trace_file.write(self.read_trace)
        self.memset_trace = ""
        self.memcpy_trace = ""
        self.read_trace = ""


# ============================================================
# Entry points
# ============================================================

def process_single_input(input_file, test_name, test_path, force=False, full_trace=False, no_read_trace=False):
    try:
        with open(test_path, 'rb') as file:
            program = pickle.load(file)

        print(f"Processing input file: {input_file}")
        program_inputs = parse_inputs(input_file)
        input_name = Path(input_file.name).stem

        trace_path = Path("positive_examples") / test_name / f"{input_name}.trace.txt"
        explored_path = Path("positive_examples") / test_name / f"{input_name}.explored_blocks.txt"
        if not force and explored_path.exists() and trace_path.exists() and trace_path.stat().st_size > 0:
            print(f"Skipping input file: {input_file} because it has already been explored")
            return

        # Clean up partial trace from a previous interrupted run
        if trace_path.exists() and not explored_path.exists():
            trace_path.unlink()

        environment = Environment(test_name, input_name)
        environment.full_trace = full_trace
        if no_read_trace:
            environment.LOG_READ = False
        interp = BoogieInterpreter(environment, program_inputs, input_name)
        interp.preprocess(program)

        entry = find_entry_point(program)
        assert entry is not None, "No entry point found"

        print(f"[process_single_input] Executing entry procedure: {entry.name}")
        interp.execute_procedure(entry)
        with open(f"positive_examples/{test_name}/{input_name}.explored_blocks.txt", "w") as f:
            for block in interp.explored_blocks:
                f.write(f"{block}\n")
    except Exception:
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Process a Boogie file.')
    arg_parser.add_argument('test_pkg_path', type=str, help='Path to the test package')
    arg_parser.add_argument('-o', '--output', type=str, help='Output file path')
    arg_parser.add_argument('--force', action='store_true', help='Force re-interpretation, ignoring caches')
    arg_parser.add_argument('--full-trace', action='store_true', help='Write full text trace (default: compact only)')
    arg_parser.add_argument('--no-read-trace', action='store_true', help='Skip read tracing (faster, writes-only trace)')
    args = arg_parser.parse_args()

    test_pkg_dir = Path(args.test_pkg_path)
    test_name = test_pkg_dir.name.removesuffix("_pkg")
    test_path = test_pkg_dir / f"{test_name}.pkl"
    assert test_path.exists(), f"Test path does not exist: {test_path}"

    # Hash-based cache: skip if inline bpl hasn't changed
    inline_bpl = Path("bpl_out") / test_name / f"{test_name}_inline.bpl"
    hash_file = Path("positive_examples") / test_name / f"{test_name}.interp.hash"
    trace_dir = Path("positive_examples") / test_name

    if inline_bpl.exists():
        bpl_hash = hashlib.sha256(inline_bpl.read_bytes()).hexdigest()
    else:
        bpl_hash = None

    # Engine mismatch check: clear traces if they were produced by a different engine
    engine_file = trace_dir / f"{test_name}.engine"
    if trace_dir.exists() and engine_file.exists():
        stored_engine = engine_file.read_text().strip()
        if stored_engine != "python":
            print(f"Engine changed ({stored_engine} → python) — clearing old traces in {trace_dir}")
            shutil.rmtree(trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            mem_trace_dir = Path("mem_ops_traces") / test_name
            if mem_trace_dir.exists():
                shutil.rmtree(mem_trace_dir)

    if not args.force and bpl_hash and hash_file.exists() and trace_dir.exists():
        stored_hash = hash_file.read_text().strip()
        if stored_hash == bpl_hash:
            # Hash matches — but check if ALL inputs are complete
            input_directory_check = Path("test_input") / f"{test_name}"
            input_files_check = list(input_directory_check.glob("*.json"))
            all_done = all(
                (trace_dir / f"{p.stem}.explored_blocks.txt").exists()
                for p in input_files_check
            )
            if all_done and input_files_check:
                print(f"Skipping interpretation — {inline_bpl} unchanged and all inputs complete")
                exit(0)
            else:
                done = sum(1 for p in input_files_check if (trace_dir / f"{p.stem}.explored_blocks.txt").exists())
                print(f"Resuming interpretation — {done}/{len(input_files_check)} inputs complete")
        elif stored_hash != bpl_hash:
            print(f"Hash mismatch — trashing old trace files in {trace_dir}")
            shutil.rmtree(trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            # Also trash old mem_ops_traces
            mem_trace_dir = Path("mem_ops_traces") / test_name
            if mem_trace_dir.exists():
                shutil.rmtree(mem_trace_dir)

    input_directory = Path("test_input") / f"{test_name}"
    input_files = list(input_directory.glob("*.json"))
    assert len(input_files) > 0, f"No input files found in {input_directory}"

    max_workers = min(max(1, os.cpu_count() - 1), len(input_files))
    print(f"Using {max_workers} workers for {len(input_files)} inputs")
    worker_func = functools.partial(process_single_input, test_name=test_name, test_path=test_path, force=args.force, full_trace=getattr(args, 'full_trace', False), no_read_trace=getattr(args, 'no_read_trace', False))

    failed = False
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(worker_func, input_files)
        try:
            for result in results_iterator:
                pass
        except Exception:
            import traceback
            print("A worker failed! Here is the stack trace:")
            traceback.print_exc()
            failed = True

    if failed:
        print("Interpretation did not complete — run again to resume")
        exit(1)

    # Write hash only after ALL inputs succeed
    if bpl_hash:
        trace_dir.mkdir(parents=True, exist_ok=True)
        hash_file.write_text(bpl_hash + "\n")

    # Record which engine produced these traces
    trace_dir.mkdir(parents=True, exist_ok=True)
    engine_file.write_text("python\n")
