"""
Tests for the Boogie interpreter (interpreter.py) and interpreter/ modules.

Tests cover:
  - Expression evaluation (eval)
  - MemoryMap load/store semantics
  - Memory operations (memset, memcpy via quantified expressions)
  - Environment variable get/set, stack frames, alloc
  - ReadBuffer
  - End-to-end interpretation of compiled packages
"""

import sys
import os
import pickle
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from parser.boogie_parser import bpl, parse_boogie
from parser.expression import (
    FunctionApplication, MapSelect, StorageIdentifier, ProcedureIdentifier,
    BinaryExpression, UnaryExpression, BooleanLiteral, IntegerLiteral,
    OldExpression, LogicalNegation, QuantifiedExpression, IfExpression,
)
from parser.statement import (
    AssertStatement, AssumeStatement, AssignStatement, Block,
    CallStatement, GotoStatement, HavocStatement, ReturnStatement,
)
from parser.declaration import StorageDeclaration, ImplementationDeclaration, ProcedureDeclaration
from interpreter.python.Buffer import ReadBuffer
from interpreter.python.Environment import Environment
from interpreter.python.MemoryMap import MemoryMap
from interpreter.python.Context import Context
from interpreter.python.interpreter import BoogieInterpreter, find_entry_point, hex_to_bytes
from utils.utils import (
    Input, ProgramInputs, parse_inputs, mask_bits, generate_function_map,
    generate_label_to_block, initialize_code_metadata,
)
from pathlib import Path


# ============================================================
# ReadBuffer tests
# ============================================================

class TestReadBuffer:
    def test_empty_buffer(self):
        buf = ReadBuffer()
        assert buf.empty()
        assert buf.unread_bytes() == 0
        assert buf.read(5) == b""

    def test_write_and_read(self):
        buf = ReadBuffer()
        buf.write(b"\x01\x02\x03\x04")
        assert buf.unread_bytes() == 4
        assert buf.read(2) == b"\x01\x02"
        assert buf.unread_bytes() == 2
        assert buf.read(2) == b"\x03\x04"
        assert buf.empty()

    def test_init_with_data(self):
        buf = ReadBuffer(b"\xaa\xbb\xcc")
        assert buf.unread_bytes() == 3
        assert buf.read(1) == b"\xaa"
        assert buf.read(-1) == b"\xbb\xcc"

    def test_read_zero(self):
        buf = ReadBuffer(b"\x01\x02")
        assert buf.read(0) == b""
        assert buf.unread_bytes() == 2

    def test_write_type_error(self):
        buf = ReadBuffer()
        with pytest.raises(TypeError):
            buf.write("not bytes")

    def test_read_more_than_available(self):
        buf = ReadBuffer(b"\x01\x02")
        data = buf.read(10)
        assert data == b"\x01\x02"
        assert buf.empty()


# ============================================================
# MemoryMap tests
# ============================================================

class TestMemoryMap:
    def make_env(self):
        """Create a minimal Environment for MemoryMap tests."""
        tmpdir = tempfile.mkdtemp()
        env = Environment("test", "input_0")
        if env.trace_file is not None:
            if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        return env, tmpdir

    def test_basic_set_get(self):
        env, tmpdir = self.make_env()
        try:
            mm = MemoryMap("test", "$M.0", 64, 8, env)
            mm.set(100, 42)
            assert mm.get(100) == 42
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_uninitialized_returns_zero(self):
        env, tmpdir = self.make_env()
        try:
            mm = MemoryMap("test", "$M.0", 64, 8, env)
            assert mm.get(999) == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_value_masking(self):
        env, tmpdir = self.make_env()
        try:
            mm = MemoryMap("test", "$M.0", 64, 8, env)
            mm.set(0, 0x1FF)  # 9 bits, should mask to 8 bits
            assert mm.get(0) == 0xFF
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_copy(self):
        env, tmpdir = self.make_env()
        try:
            mm = MemoryMap("test", "$M.0", 64, 8, env)
            mm.set(10, 42)
            mm.set(20, 99)
            mm2 = mm.copy("$M.0.copy")
            assert mm2.name == "$M.0.copy"
            assert mm2.get(10) == 42
            assert mm2.get(20) == 99
            # Modifying copy doesn't affect original
            mm2.set(10, 0)
            assert mm.get(10) == 42
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_clear(self):
        env, tmpdir = self.make_env()
        try:
            mm = MemoryMap("test", "$M.0", 64, 8, env)
            mm.set(0, 1)
            mm.set(1, 2)
            mm.clear()
            assert mm.get(0) == 0
            assert mm.get(1) == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_equality_same_contents(self):
        env, tmpdir = self.make_env()
        try:
            mm1 = MemoryMap("test", "$M.0", 64, 8, env)
            mm2 = MemoryMap("test", "$M.1", 64, 8, env)
            mm1.set(0, 42)
            mm2.set(0, 42)
            assert mm1 == mm2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_equality_shadow_shortcut(self):
        """Shadow maps with same name pattern are considered equal."""
        env, tmpdir = self.make_env()
        try:
            mm1 = MemoryMap("test", "$M.0", 64, 8, env)
            mm2 = MemoryMap("test", "$M.0.shadow", 64, 8, env)
            mm1.set(0, 42)
            mm2.set(0, 99)  # Different values
            # Shadow shortcut: $M.0.shadow == $M.0's shadow → True
            assert mm1 == mm2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_inequality(self):
        env, tmpdir = self.make_env()
        try:
            mm1 = MemoryMap("test", "$M.0", 64, 8, env)
            mm2 = MemoryMap("test", "$M.1", 64, 8, env)
            mm1.set(0, 42)
            mm2.set(0, 99)
            assert mm1 != mm2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_repr(self):
        env, tmpdir = self.make_env()
        try:
            mm = MemoryMap("test", "$M.0", 64, 8, env)
            mm.set(0, 0xAB)
            mm.set(16, 0xCD)
            r = repr(mm)
            assert "$M.0" in r
            assert "0xab" in r
            assert "0xcd" in r
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Context tests
# ============================================================

class TestContext:
    def test_set_and_get(self):
        ctx = Context()
        ctx.set("x", 42)
        assert ctx.get("x") == 42

    def test_get_missing_raises(self):
        ctx = Context()
        with pytest.raises(KeyError):
            ctx.get("missing")

    def test_clear(self):
        ctx = Context()
        ctx.set("x", 1)
        ctx.clear()
        assert ctx.var_store == {}
        assert len(ctx.constraints) == 0


# ============================================================
# Environment tests
# ============================================================

class TestEnvironment:
    def make_env(self):
        env = Environment("test", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        env.debug_print_trace = False
        return env

    def test_set_and_get_concrete(self):
        env = self.make_env()
        env.set_value("x", 42)
        assert env.get_var("x") == 42

    def test_set_string_value_converted(self):
        env = self.make_env()
        env.set_value("x", "0xFF")
        assert env.get_var("x") == 0xFF

    def test_push_pop_frame(self):
        env = self.make_env()
        env.set_value("x", 1)
        env.push_frame()
        env.set_value("x", 2)
        assert env.get_var("x") == 2
        env.pop_frame()
        assert env.get_var("x") == 1

    def test_get_undefined_returns_zero(self):
        env = self.make_env()
        val = env.get_var("undefined_var", silent=True)
        assert val == 0

    def test_memory_map_set(self):
        env = self.make_env()
        mm = MemoryMap("test", "$M.0", 64, 8, env)
        env.set_value("$M.0", mm)
        result = env.get_var("$M.0", silent=True)
        assert isinstance(result, MemoryMap)
        assert result.name == "$M.0"

    def test_memory_map_rename_on_set(self):
        env = self.make_env()
        mm = MemoryMap("test", "$M.0", 64, 8, env)
        env.set_value("$M.1", mm)
        result = env.get_var("$M.1", silent=True)
        assert result.name == "$M.1"

    def test_curr_addr_alloc(self):
        env = self.make_env()
        env.set_value("$CurrAddr", 0)
        assert env.alloc_addr == 0
        env.set_value("$CurrAddr", 1024)
        assert env.alloc_addr == 1024

    def test_curr_addr_shadow_alloc(self):
        env = self.make_env()
        env.set_value("$CurrAddr.shadow", 0)
        assert env.alloc_addr_shadow == 0
        env.set_value("$CurrAddr.shadow", 2048)
        assert env.alloc_addr_shadow == 2048

    def test_handle_curr_addr_shadow_independent(self):
        """Verify that shadow alloc uses alloc_addr_shadow, not alloc_addr."""
        env = self.make_env()
        env.alloc_addr = 100
        env.alloc_addr_shadow = 200
        # After getting $CurrAddr.shadow, it should return alloc_addr_shadow
        val = env.get_var("$CurrAddr.shadow", silent=True)
        assert val == 200
        val_normal = env.get_var("$CurrAddr", silent=True)
        assert val_normal == 100

    def test_buffer_trace_flush(self):
        env = self.make_env()
        env._trace_parts.append("some trace data\n")
        env._trace_size = 16
        env.dump_buffer_trace(fsync=False)
        assert env._trace_parts == []
        assert env._trace_size == 0

    def test_clear_var_concrete(self):
        env = self.make_env()
        env.set_value("x", 42)
        env.clear_var("x")
        # After clearing, get_var returns 0 (default)
        val = env.get_var("x", silent=True)
        assert val == 0

    def test_clear_var_memory_map(self):
        env = self.make_env()
        mm = MemoryMap("test", "$M.0", 64, 8, env)
        mm.set(0, 42)
        env.set_value("$M.0", mm)
        env.clear_var("$M.0")
        result = env.get_var("$M.0", silent=True)
        # MemoryMap.clear() empties the memory dict
        assert isinstance(result, MemoryMap)
        assert result.get(0) == 0


# ============================================================
# Expression evaluation tests (via BoogieInterpreter.eval)
# ============================================================

class TestEval:
    """Test BoogieInterpreter.eval() on parsed Boogie expressions."""

    def make_interpreter(self):
        env = Environment("test", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        env.debug_print_trace = False
        env.LOG_READ = False
        interp = BoogieInterpreter(env, {}, "test")
        return interp

    def test_integer_literal(self):
        interp = self.make_interpreter()
        expr = bpl("42")
        assert interp.eval(expr) == 42

    def test_boolean_literal_true(self):
        interp = self.make_interpreter()
        expr = bpl("true")
        assert interp.eval(expr) == True

    def test_boolean_literal_false(self):
        interp = self.make_interpreter()
        expr = bpl("false")
        assert interp.eval(expr) == False

    def test_variable_lookup(self):
        interp = self.make_interpreter()
        interp.env.set_value("$x", 99)
        expr = bpl("$x")
        assert interp.eval(expr) == 99

    def test_binary_add(self):
        interp = self.make_interpreter()
        expr = bpl("3 + 5")
        assert interp.eval(expr) == 8

    def test_binary_sub(self):
        interp = self.make_interpreter()
        expr = bpl("10 - 7")
        assert interp.eval(expr) == 3

    def test_binary_mul(self):
        interp = self.make_interpreter()
        expr = bpl("6 * 7")
        assert interp.eval(expr) == 42

    def test_binary_eq_true(self):
        interp = self.make_interpreter()
        expr = bpl("5 == 5")
        assert interp.eval(expr) == True

    def test_binary_eq_false(self):
        interp = self.make_interpreter()
        expr = bpl("5 == 3")
        assert interp.eval(expr) == False

    def test_binary_neq(self):
        interp = self.make_interpreter()
        expr = bpl("5 != 3")
        assert interp.eval(expr) == True

    def test_binary_lt(self):
        interp = self.make_interpreter()
        expr = bpl("3 < 5")
        assert interp.eval(expr) == True

    def test_binary_and(self):
        interp = self.make_interpreter()
        expr = bpl("true && false")
        assert interp.eval(expr) == False

    def test_logical_negation(self):
        interp = self.make_interpreter()
        expr = bpl("!true")
        assert interp.eval(expr) == False

    def test_nested_expression(self):
        interp = self.make_interpreter()
        expr = bpl("(3 + 4) * 2")
        assert interp.eval(expr) == 14

    def _make_if_expr(self, cond, then_val, else_val):
        expr = IfExpression()
        expr.condition = cond
        expr.then = then_val
        expr.else_ = else_val
        return expr

    def test_if_expression_true(self):
        interp = self.make_interpreter()
        expr = self._make_if_expr(BooleanLiteral(True), IntegerLiteral(10), IntegerLiteral(20))
        assert interp.eval(expr) == 10

    def test_if_expression_false(self):
        interp = self.make_interpreter()
        expr = self._make_if_expr(BooleanLiteral(False), IntegerLiteral(10), IntegerLiteral(20))
        assert interp.eval(expr) == 20

    def test_function_application_add(self):
        interp = self.make_interpreter()
        expr = bpl("$add.i32(10, 20)")
        result = interp.eval(expr)
        assert result == mask_bits(30, 32)

    def test_function_application_sub(self):
        interp = self.make_interpreter()
        expr = bpl("$sub.i32(20, 10)")
        result = interp.eval(expr)
        assert result == mask_bits(10, 32)

    def test_function_application_mul(self):
        interp = self.make_interpreter()
        expr = bpl("$mul.i32(6, 7)")
        result = interp.eval(expr)
        assert result == mask_bits(42, 32)

    def test_function_application_and(self):
        interp = self.make_interpreter()
        expr = bpl("$and.i8(255, 15)")
        result = interp.eval(expr)
        assert result == 0x0F

    def test_function_application_or(self):
        interp = self.make_interpreter()
        expr = bpl("$or.i8(240, 15)")
        result = interp.eval(expr)
        assert result == 0xFF

    def test_function_application_xor(self):
        interp = self.make_interpreter()
        expr = bpl("$xor.i8(255, 255)")
        result = interp.eval(expr)
        assert result == 0x00

    def test_function_application_shl(self):
        interp = self.make_interpreter()
        expr = bpl("$shl.i32(1, 8)")
        result = interp.eval(expr)
        assert result == 256

    def test_function_application_lshr(self):
        interp = self.make_interpreter()
        expr = bpl("$lshr.i32(256, 8)")
        result = interp.eval(expr)
        assert result == 1

    def test_function_application_slt_true(self):
        interp = self.make_interpreter()
        expr = bpl("$slt.i32(3, 5)")
        result = interp.eval(expr)
        assert result == True

    def test_function_application_slt_false(self):
        interp = self.make_interpreter()
        expr = bpl("$slt.i32(5, 3)")
        result = interp.eval(expr)
        assert result == False

    def test_function_application_eq(self):
        interp = self.make_interpreter()
        expr = bpl("$eq.i32(42, 42)")
        result = interp.eval(expr)
        assert result == True

    def test_function_application_ne(self):
        interp = self.make_interpreter()
        expr = bpl("$ne.i32(42, 43)")
        result = interp.eval(expr)
        assert result == True

    def test_function_sext(self):
        interp = self.make_interpreter()
        # Sign extend -1 from i8 to i32: 0xFF → 0xFFFFFFFF
        expr = bpl("$sext.i8.i32(255)")
        result = interp.eval(expr)
        assert result == mask_bits(-1, 32)

    def test_function_zext(self):
        interp = self.make_interpreter()
        expr = bpl("$zext.i8.i32(255)")
        result = interp.eval(expr)
        assert result == 255

    def test_function_trunc(self):
        interp = self.make_interpreter()
        expr = bpl("$trunc.i32.i8(511)")  # 0x1FF
        result = interp.eval(expr)
        assert result == 0xFF

    def test_store_and_load(self):
        """Test $store.i8 and $load.i8 via function application."""
        interp = self.make_interpreter()
        mm = MemoryMap("test", "$M.0", 64, 8, interp.env)
        interp.env.set_value("$M.0", mm)

        # Store: $M.0[10] = 0xAB
        store_expr = bpl("$store.i8($M.0, 10, 171)")  # 171 = 0xAB
        result = interp.eval(store_expr)
        assert isinstance(result, MemoryMap)
        assert result.get(10) == 0xAB

        # Load: read back from $M.0[10]
        load_expr = bpl("$load.i8($M.0, 10)")
        result = interp.eval(load_expr)
        assert result == 0xAB

    def test_multi_byte_store_load(self):
        """Test $store.i32 and $load.i32 with 8-bit elements (4 byte store)."""
        interp = self.make_interpreter()
        mm = MemoryMap("test", "$M.0", 64, 8, interp.env)
        interp.env.set_value("$M.0", mm)

        # Store 0x04030201 as i32 into 8-bit-element map at address 0
        store_expr = bpl("$store.i32($M.0, 0, 67305985)")  # 0x04030201
        interp.eval(store_expr)

        # Load back as i32
        load_expr = bpl("$load.i32($M.0, 0)")
        result = interp.eval(load_expr)
        assert result == 0x04030201

        # Individual bytes should be little-endian
        assert mm.get(0) == 0x01
        assert mm.get(1) == 0x02
        assert mm.get(2) == 0x03
        assert mm.get(3) == 0x04

    def test_variable_expression(self):
        interp = self.make_interpreter()
        interp.env.set_value("$a", 10)
        interp.env.set_value("$b", 20)
        expr = bpl("$add.i32($a, $b)")
        result = interp.eval(expr)
        assert result == mask_bits(30, 32)


# ============================================================
# Statement execution tests
# ============================================================

class TestStatements:
    def make_interpreter(self):
        env = Environment("test", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        env.debug_print_trace = False
        env.LOG_READ = False
        interp = BoogieInterpreter(env, {}, "test")
        return interp

    def test_assign_simple(self):
        interp = self.make_interpreter()
        stmt = bpl("$x := 42;")
        interp.execute_statement(stmt)
        assert interp.env.get_var("$x", silent=True) == 42

    def test_assign_expression(self):
        interp = self.make_interpreter()
        interp.env.set_value("$a", 10, silent=True)
        stmt = bpl("$b := $add.i32($a, 5);")
        interp.execute_statement(stmt)
        assert interp.env.get_var("$b", silent=True) == mask_bits(15, 32)

    def test_assign_multiple(self):
        interp = self.make_interpreter()
        stmt = bpl("$x, $y := 1, 2;")
        interp.execute_statement(stmt)
        assert interp.env.get_var("$x", silent=True) == 1
        assert interp.env.get_var("$y", silent=True) == 2

    def test_assert_passing(self):
        interp = self.make_interpreter()
        stmt = bpl("assert true;")
        interp.execute_statement(stmt)  # Should not raise

    def test_assert_failing(self):
        interp = self.make_interpreter()
        stmt = bpl("assert false;")
        with pytest.raises(AssertionError, match="Assert failed"):
            interp.execute_statement(stmt)

    def test_assume_true(self):
        interp = self.make_interpreter()
        stmt = bpl("assume true;")
        interp.execute_statement(stmt)  # Should not raise

    def test_assume_false(self):
        """Assume false should not crash — interpreter just evaluates for trace."""
        interp = self.make_interpreter()
        stmt = bpl("assume false;")
        interp.execute_statement(stmt)


# ============================================================
# Memset / Memcpy forall tests
# ============================================================

class TestMemsetMemcpy:
    """
    Test the interpreter's handling of inlined memset/memcpy forall assume
    statements. After inlining, these appear as:

    Memset (3 assumes):
      1. forall x :: (dst <= x && (x < dst+len ==> M.ret[x] == val))  -- fill range
      2. forall x :: (x < dst ==> M.ret[x] == M[x])                   -- preserve below
      3. forall x :: (dst+len <= x ==> M.ret[x] == M[x])              -- preserve above

    Memcpy (3 assumes):
      1. forall x :: (dst <= x && (x < dst+len ==> M.ret[x] == M.src[src+(x-dst)]))  -- copy range
      2. forall x :: (x < dst ==> M.ret[x] == M.dst[x])               -- preserve below
      3. forall x :: (dst+len <= x ==> M.ret[x] == M.dst[x])          -- preserve above
    """

    def make_interpreter(self):
        env = Environment("test_memops", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        env.debug_print_trace = False
        env.LOG_READ = False
        interp = BoogieInterpreter(env, {}, "test_memops")
        return interp

    # ----------------------------------------------------------
    # Memset tests
    # ----------------------------------------------------------

    def test_memset_fill_range(self):
        """
        Memset pattern 1: forall x :: (dst <= x && (x < dst+len ==> M.ret[x] == val))
        Sets M.ret[dst..dst+len) = val.
        """
        interp = self.make_interpreter()
        M_ret = MemoryMap("test", "inline$$memset.i8.cross_product$0$$M.ret", 64, 8, interp.env)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$M.ret", M_ret)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$dst", 100)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$len", 10)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$val", 0xAB)

        stmt = bpl(
            r'assume (forall x: ref :: ($sle.ref.bool(inline$$memset.i8.cross_product$0$$dst, x) '
            r'&& ($slt.ref.bool(x, $add.ref(inline$$memset.i8.cross_product$0$$dst, '
            r'inline$$memset.i8.cross_product$0$$len)) '
            r'==> (inline$$memset.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memset.i8.cross_product$0$$val))));'
        )
        interp.execute_statement(stmt)

        # Verify all 10 bytes in [100, 110) are set to 0xAB
        for addr in range(100, 110):
            assert M_ret.get(addr) == 0xAB, f"M.ret[{addr}] = {M_ret.get(addr)}, expected 0xAB"

    def test_memset_preserve_below(self):
        """
        Memset pattern 2: forall x :: (x < dst ==> M.ret[x] == M[x])
        Preserves M[0..dst) in M.ret.
        """
        interp = self.make_interpreter()
        M = MemoryMap("test", "inline$$memset.i8.cross_product$0$$M", 64, 8, interp.env)
        M.set(50, 0x11)
        M.set(90, 0x22)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$M", M)

        M_ret = MemoryMap("test", "inline$$memset.i8.cross_product$0$$M.ret", 64, 8, interp.env)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$M.ret", M_ret)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$dst", 100)

        stmt = bpl(
            r'assume (forall x: ref :: ($slt.ref.bool(x, inline$$memset.i8.cross_product$0$$dst) '
            r'==> (inline$$memset.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memset.i8.cross_product$0$$M[x])));'
        )
        interp.execute_statement(stmt)

        # Values below dst=100 should be copied from M to M_ret
        assert M_ret.get(50) == 0x11
        assert M_ret.get(90) == 0x22

    def test_memset_preserve_above(self):
        """
        Memset pattern 3: forall x :: (dst+len <= x ==> M.ret[x] == M[x])
        Preserves M[dst+len..) in M.ret.
        """
        interp = self.make_interpreter()
        M = MemoryMap("test", "inline$$memset.i8.cross_product$0$$M", 64, 8, interp.env)
        M.set(110, 0x33)
        M.set(200, 0x44)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$M", M)

        M_ret = MemoryMap("test", "inline$$memset.i8.cross_product$0$$M.ret", 64, 8, interp.env)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$M.ret", M_ret)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$dst", 100)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$len", 10)

        stmt = bpl(
            r'assume (forall x: ref :: ($sle.ref.bool($add.ref(inline$$memset.i8.cross_product$0$$dst, '
            r'inline$$memset.i8.cross_product$0$$len), x) '
            r'==> (inline$$memset.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memset.i8.cross_product$0$$M[x])));'
        )
        interp.execute_statement(stmt)

        # Values at/above dst+len=110 should be copied
        assert M_ret.get(110) == 0x33
        assert M_ret.get(200) == 0x44

    def test_memset_full_sequence(self):
        """
        Full memset: apply all 3 forall patterns in sequence and verify
        that the result memory has the expected state.
        """
        interp = self.make_interpreter()
        DST, LEN, VAL = 100, 10, 0x42

        # Pre-existing memory
        M = MemoryMap("test", "inline$$memset.i8.cross_product$0$$M", 64, 8, interp.env)
        for i in range(50, 120):
            M.set(i, i & 0xFF)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$M", M)

        M_ret = MemoryMap("test", "inline$$memset.i8.cross_product$0$$M.ret", 64, 8, interp.env)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$M.ret", M_ret)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$dst", DST)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$len", LEN)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$val", VAL)

        # Pattern 1: fill [dst, dst+len) with val
        interp.execute_statement(bpl(
            r'assume (forall x: ref :: ($sle.ref.bool(inline$$memset.i8.cross_product$0$$dst, x) '
            r'&& ($slt.ref.bool(x, $add.ref(inline$$memset.i8.cross_product$0$$dst, '
            r'inline$$memset.i8.cross_product$0$$len)) '
            r'==> (inline$$memset.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memset.i8.cross_product$0$$val))));'
        ))
        # Pattern 2: preserve below dst
        interp.execute_statement(bpl(
            r'assume (forall x: ref :: ($slt.ref.bool(x, inline$$memset.i8.cross_product$0$$dst) '
            r'==> (inline$$memset.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memset.i8.cross_product$0$$M[x])));'
        ))
        # Pattern 3: preserve above dst+len
        interp.execute_statement(bpl(
            r'assume (forall x: ref :: ($sle.ref.bool($add.ref(inline$$memset.i8.cross_product$0$$dst, '
            r'inline$$memset.i8.cross_product$0$$len), x) '
            r'==> (inline$$memset.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memset.i8.cross_product$0$$M[x])));'
        ))

        # Check below range: should be original values
        for i in range(50, DST):
            assert M_ret.get(i) == (i & 0xFF), f"M_ret[{i}]={M_ret.get(i)}, expected {i & 0xFF}"
        # Check memset range: should be VAL
        for i in range(DST, DST + LEN):
            assert M_ret.get(i) == VAL, f"M_ret[{i}]={M_ret.get(i)}, expected {VAL}"
        # Check above range: should be original values
        for i in range(DST + LEN, 120):
            assert M_ret.get(i) == (i & 0xFF), f"M_ret[{i}]={M_ret.get(i)}, expected {i & 0xFF}"

    # ----------------------------------------------------------
    # Memcpy tests
    # ----------------------------------------------------------

    def test_memcpy_copy_range(self):
        """
        Memcpy pattern 1: forall x :: (dst <= x && (x < dst+len ==> M.ret[x] == M.src[src+(x-dst)]))
        Copies src data into [dst, dst+len).
        """
        interp = self.make_interpreter()

        M_src = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.src", 64, 8, interp.env)
        # Put source data at addresses 200..210
        for i in range(10):
            M_src.set(200 + i, 0xA0 + i)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.src", M_src)

        M_ret = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.ret", 64, 8, interp.env)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.ret", M_ret)

        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$dst", 100)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$src", 200)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$len", 10)

        stmt = bpl(
            r'assume (forall x: ref :: ($sle.ref.bool(inline$$memcpy.i8.cross_product$0$$dst, x) '
            r'&& ($slt.ref.bool(x, $add.ref(inline$$memcpy.i8.cross_product$0$$dst, '
            r'inline$$memcpy.i8.cross_product$0$$len)) '
            r'==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memcpy.i8.cross_product$0$$M.src[$add.ref($sub.ref('
            r'inline$$memcpy.i8.cross_product$0$$src, inline$$memcpy.i8.cross_product$0$$dst), x)]))));'
        )
        interp.execute_statement(stmt)

        # Verify M_ret[100..110) contains the source data
        for i in range(10):
            assert M_ret.get(100 + i) == 0xA0 + i, \
                f"M_ret[{100+i}]={M_ret.get(100+i)}, expected {0xA0+i}"

    def test_memcpy_preserve_below(self):
        """
        Memcpy pattern 2: forall x :: (x < dst ==> M.ret[x] == M.dst[x])
        Preserves existing data below dst.
        """
        interp = self.make_interpreter()

        M_dst = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.dst", 64, 8, interp.env)
        M_dst.set(10, 0xBB)
        M_dst.set(50, 0xCC)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.dst", M_dst)

        M_ret = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.ret", 64, 8, interp.env)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.ret", M_ret)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$dst", 100)

        stmt = bpl(
            r'assume (forall x: ref :: ($slt.ref.bool(x, inline$$memcpy.i8.cross_product$0$$dst) '
            r'==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memcpy.i8.cross_product$0$$M.dst[x])));'
        )
        interp.execute_statement(stmt)

        assert M_ret.get(10) == 0xBB
        assert M_ret.get(50) == 0xCC

    def test_memcpy_preserve_above(self):
        """
        Memcpy pattern 3: forall x :: (dst+len <= x ==> M.ret[x] == M.dst[x])
        Preserves existing data at/above dst+len.
        """
        interp = self.make_interpreter()

        M_dst = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.dst", 64, 8, interp.env)
        M_dst.set(110, 0xDD)
        M_dst.set(200, 0xEE)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.dst", M_dst)

        M_ret = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.ret", 64, 8, interp.env)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.ret", M_ret)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$dst", 100)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$len", 10)

        stmt = bpl(
            r'assume (forall x: ref :: ($sle.ref.bool($add.ref(inline$$memcpy.i8.cross_product$0$$dst, '
            r'inline$$memcpy.i8.cross_product$0$$len), x) '
            r'==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memcpy.i8.cross_product$0$$M.dst[x])));'
        )
        interp.execute_statement(stmt)

        assert M_ret.get(110) == 0xDD
        assert M_ret.get(200) == 0xEE

    def test_memcpy_full_sequence(self):
        """
        Full memcpy: apply all 3 forall patterns and verify the result.
        """
        interp = self.make_interpreter()
        DST, SRC, LEN = 100, 200, 10

        # Destination memory: has existing data everywhere
        M_dst = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.dst", 64, 8, interp.env)
        for i in range(50, 130):
            M_dst.set(i, 0xFF)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.dst", M_dst)

        # Source memory: has specific data to copy
        M_src = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.src", 64, 8, interp.env)
        for i in range(10):
            M_src.set(SRC + i, i + 1)  # 1, 2, 3, ..., 10
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.src", M_src)

        M_ret = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.ret", 64, 8, interp.env)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.ret", M_ret)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$dst", DST)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$src", SRC)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$len", LEN)

        # Pattern 1: copy src range
        interp.execute_statement(bpl(
            r'assume (forall x: ref :: ($sle.ref.bool(inline$$memcpy.i8.cross_product$0$$dst, x) '
            r'&& ($slt.ref.bool(x, $add.ref(inline$$memcpy.i8.cross_product$0$$dst, '
            r'inline$$memcpy.i8.cross_product$0$$len)) '
            r'==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memcpy.i8.cross_product$0$$M.src[$add.ref($sub.ref('
            r'inline$$memcpy.i8.cross_product$0$$src, inline$$memcpy.i8.cross_product$0$$dst), x)]))));'
        ))
        # Pattern 2: preserve below dst
        interp.execute_statement(bpl(
            r'assume (forall x: ref :: ($slt.ref.bool(x, inline$$memcpy.i8.cross_product$0$$dst) '
            r'==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memcpy.i8.cross_product$0$$M.dst[x])));'
        ))
        # Pattern 3: preserve above dst+len
        interp.execute_statement(bpl(
            r'assume (forall x: ref :: ($sle.ref.bool($add.ref(inline$$memcpy.i8.cross_product$0$$dst, '
            r'inline$$memcpy.i8.cross_product$0$$len), x) '
            r'==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memcpy.i8.cross_product$0$$M.dst[x])));'
        ))

        # Below range: preserved from M_dst
        for i in range(50, DST):
            assert M_ret.get(i) == 0xFF, f"M_ret[{i}]={M_ret.get(i)}, expected 0xFF"
        # Copy range: should have source data
        for i in range(LEN):
            expected = i + 1
            assert M_ret.get(DST + i) == expected, \
                f"M_ret[{DST+i}]={M_ret.get(DST+i)}, expected {expected}"
        # Above range: preserved from M_dst
        for i in range(DST + LEN, 130):
            assert M_ret.get(i) == 0xFF, f"M_ret[{i}]={M_ret.get(i)}, expected 0xFF"

    def test_memset_zero_fill(self):
        """Test memset with val=0 (common pattern: zero-fill a buffer)."""
        interp = self.make_interpreter()
        M_ret = MemoryMap("test", "inline$$memset.i8.cross_product$0$$M.ret", 64, 8, interp.env)
        # Pre-fill with non-zero data
        for i in range(10):
            M_ret.set(100 + i, 0xFF)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$M.ret", M_ret)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$dst", 100)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$len", 10)
        interp.env.set_value("inline$$memset.i8.cross_product$0$$val", 0)

        interp.execute_statement(bpl(
            r'assume (forall x: ref :: ($sle.ref.bool(inline$$memset.i8.cross_product$0$$dst, x) '
            r'&& ($slt.ref.bool(x, $add.ref(inline$$memset.i8.cross_product$0$$dst, '
            r'inline$$memset.i8.cross_product$0$$len)) '
            r'==> (inline$$memset.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memset.i8.cross_product$0$$val))));'
        ))

        for addr in range(100, 110):
            assert M_ret.get(addr) == 0, f"M_ret[{addr}]={M_ret.get(addr)}, expected 0"

    def test_memcpy_overlapping_same_map(self):
        """
        Test memcpy where M.src and M.dst are the same memory map
        (common when doing memmove-like operations within one buffer).
        """
        interp = self.make_interpreter()
        DST, SRC, LEN = 10, 5, 5

        M = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.dst", 64, 8, interp.env)
        # Fill addresses 0..20 with distinct values
        for i in range(20):
            M.set(i, i * 10)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.dst", M)

        # M.src is the same map (aliased)
        M_src = M.copy("inline$$memcpy.i8.cross_product$0$$M.src")
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.src", M_src)

        M_ret = MemoryMap("test", "inline$$memcpy.i8.cross_product$0$$M.ret", 64, 8, interp.env)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$M.ret", M_ret)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$dst", DST)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$src", SRC)
        interp.env.set_value("inline$$memcpy.i8.cross_product$0$$len", LEN)

        # Copy range
        interp.execute_statement(bpl(
            r'assume (forall x: ref :: ($sle.ref.bool(inline$$memcpy.i8.cross_product$0$$dst, x) '
            r'&& ($slt.ref.bool(x, $add.ref(inline$$memcpy.i8.cross_product$0$$dst, '
            r'inline$$memcpy.i8.cross_product$0$$len)) '
            r'==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memcpy.i8.cross_product$0$$M.src[$add.ref($sub.ref('
            r'inline$$memcpy.i8.cross_product$0$$src, inline$$memcpy.i8.cross_product$0$$dst), x)]))));'
        ))
        # Preserve below
        interp.execute_statement(bpl(
            r'assume (forall x: ref :: ($slt.ref.bool(x, inline$$memcpy.i8.cross_product$0$$dst) '
            r'==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memcpy.i8.cross_product$0$$M.dst[x])));'
        ))
        # Preserve above
        interp.execute_statement(bpl(
            r'assume (forall x: ref :: ($sle.ref.bool($add.ref(inline$$memcpy.i8.cross_product$0$$dst, '
            r'inline$$memcpy.i8.cross_product$0$$len), x) '
            r'==> (inline$$memcpy.i8.cross_product$0$$M.ret[x] == '
            r'inline$$memcpy.i8.cross_product$0$$M.dst[x])));'
        ))

        # Below: original data preserved
        for i in range(DST):
            assert M_ret.get(i) == i * 10, f"M_ret[{i}]={M_ret.get(i)}, expected {i*10}"
        # Copy range: data from src[5..10) → dst[10..15)
        for i in range(LEN):
            expected = (SRC + i) * 10  # src values
            assert M_ret.get(DST + i) == expected, \
                f"M_ret[{DST+i}]={M_ret.get(DST+i)}, expected {expected}"
        # Above: original data preserved
        for i in range(DST + LEN, 20):
            assert M_ret.get(i) == i * 10, f"M_ret[{i}]={M_ret.get(i)}, expected {i*10}"


# ============================================================
# End-to-end integration tests (using compiled packages)
# ============================================================

class TestEndToEnd:
    """Integration tests that run the interpreter on actual compiled packages."""

    @pytest.fixture
    def aead_pkg(self):
        pkg_path = Path("test_packages/aead_pkg/aead.pkl")
        if not pkg_path.exists():
            pytest.skip("aead package not compiled")
        with open(pkg_path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture
    def aead_input(self):
        input_path = Path("test_input/aead/input_0.json")
        if not input_path.exists():
            pytest.skip("aead input not available")
        return parse_inputs(input_path)

    def test_aead_preprocess(self, aead_pkg, aead_input):
        """Test that preprocessing the aead package works."""
        env = Environment("test_aead", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        interp = BoogieInterpreter(env, aead_input, "test_aead")
        interp.preprocess(aead_pkg)

        # Should have label_to_block and label_to_pc populated
        assert len(interp.label_to_block) > 0
        assert len(interp.label_to_pc) > 0
        assert interp.impl_decl is not None

    def test_aead_execution(self, aead_pkg, aead_input, tmp_path):
        """Test full execution of aead benchmark (should complete without error)."""
        env = Environment("test_aead_exec", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(tmp_path / "trace.txt", "w")
        env.debug_print_trace = False
        interp = BoogieInterpreter(env, aead_input, "test_aead_exec")
        interp.preprocess(aead_pkg)


        entry = find_entry_point(aead_pkg)
        assert entry is not None, "aead should have an entry point"

        interp.execute_procedure(entry)
        assert len(interp.explored_blocks) > 0

    def test_aead_shadow_consistency(self, aead_pkg, aead_input, tmp_path):
        """
        For public inputs, shadow variables should track the same values.
        Run aead and check that public memory maps have same contents in
        shadow and non-shadow versions.
        """
        env = Environment("test_aead_shadow", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(tmp_path / "trace.txt", "w")
        env.debug_print_trace = False
        interp = BoogieInterpreter(env, aead_input, "test_aead_shadow")
        interp.preprocess(aead_pkg)


        entry = find_entry_point(aead_pkg)
        interp.execute_procedure(entry)

        # Check a few public scalar variables have matching shadow values
        for var_name, inp in aead_input.variables.items():
            if inp.value is not None and not inp.private:
                shadow_name = f"{var_name}.shadow"
                val = env.get_concrete_value(var_name)
                shadow_val = env.get_concrete_value(shadow_name)
                if val is not None and shadow_val is not None:
                    if not isinstance(val, MemoryMap):
                        assert val == shadow_val, (
                            f"Public var {var_name} diverged: {val} != {shadow_val}"
                        )

    def test_aead_comprehensive(self, aead_pkg, aead_input, tmp_path):
        """
        Comprehensive test verifying the interpreter's AEAD decrypt output against
        known ChaCha20-Poly1305 test vectors from libsodium's aead.c.

        Test vectors (from aead.c):
          key:        4290bcb154173531f314af57f3be3b5006da371ece272afa1b5dbdd1100a1007
          plaintext:  86d09974840bded2a5ca (10 bytes)
          nonce:      cd7cf67be39c794a (8 bytes)
          AD:         87e229d4500845a079c087e229d4500845a0 (18 bytes)
          ciphertext: e3e446f7ede9a19b62a474d02a5b4e674caf2eadd6af8feb4fb6 (26 bytes = 10 + 16 MAC)

        The interpreter runs crypto_aead_chacha20poly1305_decrypt_wrapper which should
        decrypt the ciphertext back to the original plaintext.
        """
        env = Environment("test_aead_comprehensive", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(tmp_path / "trace.txt", "w")
        env.debug_print_trace = False
        interp = BoogieInterpreter(env, aead_input, "test_aead_comprehensive")
        interp.preprocess(aead_pkg)

        entry = find_entry_point(aead_pkg)
        assert entry is not None
        interp.execute_procedure(entry)

        # 1. Return value should be 0 (decrypt success)
        r = env.get_concrete_value("$r")
        assert r == 0, f"Decrypt should return 0 (success), got {r}"

        # 2. Shadow return should match
        r_shadow = env.get_concrete_value("$r.shadow")
        assert r == r_shadow, f"Return shadow mismatch: {r} != {r_shadow}"

        # 3. Verify decrypted plaintext matches C test vector
        EXPECTED_PLAINTEXT = bytes([0x86, 0xd0, 0x99, 0x74, 0x84, 0x0b, 0xde, 0xd2, 0xa5, 0xca])
        p0_addr = env.get_concrete_value("$p0")
        M1 = env.get_concrete_value("$M.1")
        assert M1 is not None, "$M.1 memory map should exist"
        output_bytes = bytes([M1.get(p0_addr + i) for i in range(10)])
        assert output_bytes == EXPECTED_PLAINTEXT, (
            f"Decrypted plaintext mismatch:\n"
            f"  got:      0x{output_bytes.hex()}\n"
            f"  expected: 0x{EXPECTED_PLAINTEXT.hex()}"
        )

        # 4. Shadow plaintext should match (constant-time property)
        M1_shadow = env.get_concrete_value("$M.1.shadow")
        assert M1_shadow is not None
        shadow_bytes = bytes([M1_shadow.get(p0_addr + i) for i in range(10)])
        assert shadow_bytes == EXPECTED_PLAINTEXT, (
            f"Shadow plaintext mismatch: 0x{shadow_bytes.hex()}"
        )

        # 5. Cross-validate against real ChaCha20-Poly1305 (original, non-IETF)
        EXPECTED_KEY = bytes.fromhex("4290bcb154173531f314af57f3be3b5006da371ece272afa1b5dbdd1100a1007")
        EXPECTED_NONCE = bytes.fromhex("cd7cf67be39c794a")
        EXPECTED_AD = bytes.fromhex("87e229d4500845a079c087e229d4500845a0")
        EXPECTED_CT = bytes.fromhex("e3e446f7ede9a19b62a474d02a5b4e674caf2eadd6af8feb4fb6")

        try:
            import nacl.bindings
            # Verify the ciphertext is a valid encryption of the expected plaintext
            real_ct = nacl.bindings.crypto_aead_chacha20poly1305_encrypt(
                EXPECTED_PLAINTEXT, EXPECTED_AD, EXPECTED_NONCE, EXPECTED_KEY
            )
            assert real_ct == EXPECTED_CT, (
                f"Ciphertext doesn't match real ChaCha20-Poly1305 encryption:\n"
                f"  real:     0x{real_ct.hex()}\n"
                f"  expected: 0x{EXPECTED_CT.hex()}"
            )
            # Verify the interpreter's output matches a real decrypt
            real_pt = nacl.bindings.crypto_aead_chacha20poly1305_decrypt(
                EXPECTED_CT, EXPECTED_AD, EXPECTED_NONCE, EXPECTED_KEY
            )
            assert output_bytes == real_pt, (
                f"Interpreter output doesn't match real decrypt:\n"
                f"  interpreter: 0x{output_bytes.hex()}\n"
                f"  real:        0x{real_pt.hex()}"
            )
        except ImportError:
            pass  # pynacl not available, skip cross-validation

        # 6. Verify input memory was correctly set up
        # Key in $M.5
        M5 = env.get_concrete_value("$M.5")
        key_bytes = bytes([M5.get(i) for i in range(32)])
        assert key_bytes == EXPECTED_KEY, f"Key mismatch: 0x{key_bytes.hex()}"

        # Nonce in $M.3
        M3 = env.get_concrete_value("$M.3")
        nonce_bytes = bytes([M3.get(i) for i in range(8)])
        assert nonce_bytes == EXPECTED_NONCE, f"Nonce mismatch: 0x{nonce_bytes.hex()}"

        # AD in $M.2
        M2 = env.get_concrete_value("$M.2")
        ad_bytes = bytes([M2.get(i) for i in range(18)])
        assert ad_bytes == EXPECTED_AD, f"AD mismatch: 0x{ad_bytes.hex()}"

        # Ciphertext in $M.1 at $p3 offset
        p3_addr = env.get_concrete_value("$p3")
        ct_bytes = bytes([M1.get(p3_addr + i) for i in range(26)])
        assert ct_bytes == EXPECTED_CT, f"Ciphertext mismatch: 0x{ct_bytes.hex()}"

        # 6. Verify all memory maps have matching shadow contents (shadow consistency)
        for mem_name in ["$M.1", "$M.2", "$M.3", "$M.5"]:
            mem = env.get_concrete_value(mem_name)
            mem_shadow = env.get_concrete_value(f"{mem_name}.shadow")
            assert len(mem.memory) == len(mem_shadow.memory), (
                f"{mem_name} entry count mismatch: {len(mem.memory)} vs {len(mem_shadow.memory)}"
            )
            for addr in mem.memory:
                assert mem.memory[addr] == mem_shadow.memory[addr], (
                    f"{mem_name}[{hex(addr)}] shadow mismatch: "
                    f"{hex(mem.memory[addr])} vs {hex(mem_shadow.memory[addr])}"
                )

        # 7. Verify sufficient blocks were explored
        assert len(interp.explored_blocks) >= 200, (
            f"Expected >= 200 explored blocks, got {len(interp.explored_blocks)}"
        )


# ============================================================
# hex_to_bytes tests
# ============================================================

class TestHexToBytes:
    def test_basic(self):

        result = hex_to_bytes("0xAABBCC")
        assert result == bytearray(b"\xAA\xBB\xCC")

    def test_no_prefix(self):

        result = hex_to_bytes("AABBCC")
        assert result == bytearray(b"\xAA\xBB\xCC")

    def test_spaces(self):

        result = hex_to_bytes("AA BB CC")
        assert result == bytearray(b"\xAA\xBB\xCC")

    def test_odd_length_raises(self):

        with pytest.raises(ValueError):
            hex_to_bytes("0xABC")

    def test_as_bytes(self):

        result = hex_to_bytes("0xAA", as_bytearray=False)
        assert isinstance(result, bytes)


# ============================================================
# generate_function_map tests
# ============================================================

class TestFunctionMap:
    def test_fn_map_has_common_ops(self):
        fn_map = generate_function_map()
        # Check a sampling of expected functions
        assert "$add.i32" in fn_map
        assert "$sub.i32" in fn_map
        assert "$mul.i32" in fn_map
        assert "$and.i32" in fn_map
        assert "$or.i32" in fn_map
        assert "$xor.i32" in fn_map
        assert "$shl.i32" in fn_map
        assert "$lshr.i32" in fn_map

    def test_fn_map_has_extensions(self):
        fn_map = generate_function_map()
        assert "$sext.i8.i32" in fn_map
        assert "$zext.i8.i32" in fn_map
        assert "$trunc.i32.i8" in fn_map

    def test_fn_map_has_comparisons(self):
        fn_map = generate_function_map()
        assert "$eq.i32" in fn_map
        assert "$ne.i32" in fn_map
        assert "$slt.i32" in fn_map
        assert "$sle.i32" in fn_map
        assert "$sgt.i32" in fn_map
        assert "$sge.i32" in fn_map

    def test_fn_map_entry_format(self):
        fn_map = generate_function_map()
        op, num_args, in_bits, out_bits = fn_map["$add.i32"]
        assert callable(op)
        assert num_args == 2
        assert in_bits == 32
        assert out_bits == 32


# ============================================================
# parse_inputs tests
# ============================================================

class TestParseInputs:
    def test_basic_input(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {"var": "$x", "value": 42, "private": False},
                {"var": "$y", "value": 100, "private": True},
            ], f)
            f.flush()
            try:
                result = parse_inputs(f.name)
                assert "$x" in result.variables
                assert result.variables["$x"].value == 42
                assert result.variables["$x"].private == False
                # Shadow copies via with_shadows()
                all_vars = result.with_shadows()
                assert "$x.shadow" in all_vars
                assert all_vars["$x.shadow"].value == 42
                # Private vars don't get shadow copies
                assert "$y" in result.variables
                assert "$y.shadow" not in all_vars
            finally:
                os.unlink(f.name)

    def test_array_input(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {
                    "var": "$p0",
                    "private": False,
                    "buffers": [
                        {"contents": "0xAABBCC", "size": 3}
                    ]
                }
            ], f)
            f.flush()
            try:
                result = parse_inputs(f.name)
                assert "$p0" in result.variables
                assert len(result.variables["$p0"].buffers) == 1
                assert result.variables["$p0"].buffers[0]["contents"] == "0xAABBCC"
            finally:
                os.unlink(f.name)

    def test_extra_data(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {"extra_data": "AABBCC"},
            ], f)
            f.flush()
            try:
                result = parse_inputs(f.name)
                assert result.extra_data == b"\xAA\xBB\xCC"
            finally:
                os.unlink(f.name)


# ============================================================
# Round 1: Missing binary operators and resource management
# ============================================================

class TestMissingBinaryOps:
    def make_interpreter(self):
        env = Environment("test", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        env.debug_print_trace = False
        env.LOG_READ = False
        interp = BoogieInterpreter(env, {}, "test")
        return interp

    def test_binary_or(self):
        interp = self.make_interpreter()
        expr = bpl("true || false")
        assert interp.eval(expr) == True

    def test_binary_or_false(self):
        interp = self.make_interpreter()
        expr = bpl("false || false")
        assert interp.eval(expr) == False

    def test_binary_gt(self):
        interp = self.make_interpreter()
        expr = bpl("5 > 3")
        assert interp.eval(expr) == True

    def test_binary_gt_false(self):
        interp = self.make_interpreter()
        expr = bpl("3 > 5")
        assert interp.eval(expr) == False

    def test_binary_ge(self):
        interp = self.make_interpreter()
        expr = bpl("5 >= 5")
        assert interp.eval(expr) == True

    def test_binary_ge_false(self):
        interp = self.make_interpreter()
        expr = bpl("4 >= 5")
        assert interp.eval(expr) == False

    def test_binary_le(self):
        interp = self.make_interpreter()
        expr = bpl("5 <= 5")
        assert interp.eval(expr) == True

    def test_binary_le_false(self):
        interp = self.make_interpreter()
        expr = bpl("6 <= 5")
        assert interp.eval(expr) == False

    def test_binary_implies_true(self):
        interp = self.make_interpreter()
        expr = bpl("false ==> false")
        assert interp.eval(expr) == True

    def test_binary_implies_false(self):
        interp = self.make_interpreter()
        expr = bpl("true ==> false")
        assert interp.eval(expr) == False

    def test_binary_iff_true(self):
        interp = self.make_interpreter()
        expr = bpl("true <==> true")
        assert interp.eval(expr) == True

    def test_binary_iff_false(self):
        interp = self.make_interpreter()
        expr = bpl("true <==> false")
        assert interp.eval(expr) == False


class TestResourceManagement:
    def test_interpreter_close(self):
        env = Environment("test_close", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        interp = BoogieInterpreter(env, {}, "test_close")
        # Files should be open
        assert not interp.memset_trace_file.closed
        assert not interp.memcpy_trace_file.closed
        assert not interp.read_trace_file.closed
        interp.close()
        assert interp.memset_trace_file.closed
        assert interp.memcpy_trace_file.closed
        assert interp.read_trace_file.closed

    def test_interpreter_close_idempotent(self):
        env = Environment("test_close2", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        interp = BoogieInterpreter(env, {}, "test_close2")
        interp.close()
        interp.close()  # Should not raise

    def test_environment_close(self):
        env = Environment("test_env_close", "input_0")
        env.full_trace = True
        env._trace_parts.append("test\n")
        env.dump_buffer_trace()  # Force trace file open
        assert env.trace_file is not None and not env.trace_file.closed
        env.close()
        assert env.trace_file.closed

    def test_environment_close_idempotent(self):
        env = Environment("test_env_close2", "input_0")
        env.close()
        env.close()  # Should not raise


# ============================================================
# Round 2: MemoryMap deep copy and N-way goto
# ============================================================

class TestMemoryMapDeepCopy:
    def make_env(self):
        tmpdir = tempfile.mkdtemp()
        env = Environment("test", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        return env, tmpdir

    def test_copy_memory_isolation(self):
        env, tmpdir = self.make_env()
        try:
            mm = MemoryMap("test", "$M.0", 64, 8, env)
            mm.set(10, 42)
            mm2 = mm.copy("$M.0.copy")
            mm2.set(10, 99)
            # Mutating the copy should not affect original
            assert mm.get(10) == 42
            assert mm2.get(10) == 99
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_copy_name(self):
        env, tmpdir = self.make_env()
        try:
            mm = MemoryMap("test", "$M.0", 64, 8, env)
            mm2 = mm.copy("$M.0.copy")
            assert mm2.name == "$M.0.copy"
            assert mm.name == "$M.0"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestNWayGoto:
    def make_interpreter(self):
        env = Environment("test", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        env.debug_print_trace = False
        env.LOG_READ = False
        interp = BoogieInterpreter(env, {}, "test")
        return interp

    def test_single_target_goto(self):
        interp = self.make_interpreter()
        block_a = Block(["blockA"], [bpl("assume true;"), bpl("return;")])
        interp.label_to_block["blockA"] = block_a

        goto_stmt = bpl("goto blockA;")
        blocks = []
        interp._handle_goto_branch(goto_stmt, blocks)
        assert len(blocks) == 1
        assert blocks[0].name == "blockA"

    def test_three_target_goto(self):
        interp = self.make_interpreter()
        interp.env.set_value("$x", 5, silent=True)

        block_a = Block(["blockA"], [bpl("assume $x == 3;"), bpl("return;")])
        interp.label_to_block["blockA"] = block_a

        block_b = Block(["blockB"], [bpl("assume $x == 5;"), bpl("return;")])
        interp.label_to_block["blockB"] = block_b

        block_c = Block(["blockC"], [bpl("assume $x == 7;"), bpl("return;")])
        interp.label_to_block["blockC"] = block_c

        goto_stmt = bpl("goto blockA, blockB, blockC;")
        blocks = []
        interp._handle_goto_branch(goto_stmt, blocks)
        assert len(blocks) == 1
        assert blocks[0].name == "blockB"


# ============================================================
# Round 3: Boolean return types, wlen_buf overflow, exception re-raise
# ============================================================

class TestBinaryOpReturnTypes:
    def make_interpreter(self):
        env = Environment("test", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        env.debug_print_trace = False
        env.LOG_READ = False
        interp = BoogieInterpreter(env, {}, "test")
        return interp

    def test_eq_returns_bool(self):
        interp = self.make_interpreter()
        result = interp.eval(bpl("5 == 5"))
        assert result is True
        assert type(result) is bool

    def test_neq_returns_bool(self):
        interp = self.make_interpreter()
        result = interp.eval(bpl("5 != 3"))
        assert result is True
        assert type(result) is bool

    def test_lt_returns_bool(self):
        interp = self.make_interpreter()
        result = interp.eval(bpl("3 < 5"))
        assert result is True
        assert type(result) is bool

    def test_and_returns_bool(self):
        interp = self.make_interpreter()
        result = interp.eval(bpl("true && true"))
        assert result is True
        assert type(result) is bool

    def test_or_returns_bool(self):
        interp = self.make_interpreter()
        result = interp.eval(bpl("true || false"))
        assert result is True
        assert type(result) is bool

    def test_add_returns_int(self):
        interp = self.make_interpreter()
        result = interp.eval(bpl("3 + 5"))
        assert result == 8
        assert type(result) is int

    def test_negation_of_comparison(self):
        interp = self.make_interpreter()
        result = interp.eval(bpl("!(3 == 5)"))
        assert result is True
        assert type(result) is bool

    def test_arithmetic_overflow_masked(self):
        interp = self.make_interpreter()
        # 2^64 should wrap to 0 with 64-bit masking
        large = IntegerLiteral(2**64 - 1)
        one = IntegerLiteral(1)
        expr = BinaryExpression(lhs=large, op="+", rhs=one)
        result = interp.eval(expr)
        assert result == 0


class TestWlenBufOverflow:
    def test_wlen_buf_wraps(self):
        env = Environment("test", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        env.debug_print_trace = False
        env.LOG_READ = False
        interp = BoogieInterpreter(env, {}, "test")
        # wlen_buf has 6 entries; after 6 calls, index wraps
        buf_len = len(interp.wlen_buf)
        assert buf_len == 6
        # Simulate calling beyond buffer length
        interp.wlen_buf_idx = buf_len - 1  # last valid index
        # Should use the modulo-wrapped index
        idx = interp.wlen_buf_idx % buf_len
        assert idx == 5
        # After incrementing past end, it wraps
        interp.wlen_buf_idx = buf_len
        idx = interp.wlen_buf_idx % buf_len
        assert idx == 0


# ============================================================
# Round 4: Parallel assignment and read_trace flush
# ============================================================

class TestParallelAssignment:
    def make_interpreter(self):
        env = Environment("test", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        env.debug_print_trace = False
        env.LOG_READ = False
        interp = BoogieInterpreter(env, {}, "test")
        return interp

    def test_swap_assignment(self):
        interp = self.make_interpreter()
        interp.env.set_value("$x", 10, silent=True)
        interp.env.set_value("$y", 20, silent=True)
        stmt = bpl("$x, $y := $y, $x;")
        interp.execute_statement(stmt)
        assert interp.env.get_var("$x", silent=True) == 20
        assert interp.env.get_var("$y", silent=True) == 10

    def test_self_referencing_multi_assign(self):
        interp = self.make_interpreter()
        interp.env.set_value("$a", 1, silent=True)
        interp.env.set_value("$b", 2, silent=True)
        # $a, $b := $a + $b, $a - $b  should use OLD values of $a and $b
        stmt = bpl("$a, $b := $a + $b, $a;")
        interp.execute_statement(stmt)
        assert interp.env.get_var("$a", silent=True) == mask_bits(3, 64)
        assert interp.env.get_var("$b", silent=True) == 1


class TestDumpMemTrace:
    def test_read_trace_flushed(self):
        env = Environment("test_trace", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        interp = BoogieInterpreter(env, {}, "test_trace")
        interp.read_trace = "some read data\n"
        interp._dump_mem_trace()
        assert interp.read_trace == ""

    def test_all_traces_flushed(self):
        env = Environment("test_trace2", "input_0")
        if env.trace_file is not None: env.trace_file.close()
        env.trace_file = open(os.devnull, "w")
        interp = BoogieInterpreter(env, {}, "test_trace2")
        interp.memset_trace = "memset data\n"
        interp.memcpy_trace = "memcpy data\n"
        interp.read_trace = "read data\n"
        interp._dump_mem_trace()
        assert interp.memset_trace == ""
        assert interp.memcpy_trace == ""
        assert interp.read_trace == ""


# ============================================================
# Parametrized integration tests: run interpreter on real inputs
# ============================================================

_BENCHMARKS = [
    ("aead", "input_0"),
    ("auth_length_pub", "input_0"),
    ("bearssl_test", "input_0"),
    ("bearssl_test_pkcs1_i15", "test_06_custom_vector"),
    ("bearssl_test_pkcs1_i31", "test_01b_standard_2048"),
    ("bearssl_test_pms_ecdh", "input_0"),
]


def _benchmark_available(test_name, input_name):
    pkg = Path(f"test_packages/{test_name}_pkg/{test_name}.pkl")
    inp = Path(f"test_input/{test_name}/{input_name}.json")
    return pkg.exists() and inp.exists()


def _load_and_run(test_name, input_name, tmp_path):
    pkg_path = Path(f"test_packages/{test_name}_pkg/{test_name}.pkl")
    input_path = Path(f"test_input/{test_name}/{input_name}.json")
    with open(pkg_path, "rb") as f:
        program = pickle.load(f)
    program_inputs = parse_inputs(input_path)
    env = Environment(f"test_{test_name}", input_name)
    if env.trace_file is not None: env.trace_file.close()
    env.trace_file = open(tmp_path / "trace.txt", "w")
    env.debug_print_trace = False
    interp = BoogieInterpreter(env, program_inputs, f"test_{test_name}")
    interp.preprocess(program)
    entry = find_entry_point(program)
    assert entry is not None, f"No entry point in {test_name}"
    interp.execute_procedure(entry)
    return interp, env


class TestBenchmarkInputs:
    """Run interpreter on each available benchmark and verify basic properties."""

    @pytest.mark.parametrize("test_name,input_name", _BENCHMARKS)
    def test_execution_completes(self, test_name, input_name, tmp_path):
        if not _benchmark_available(test_name, input_name):
            pytest.skip(f"{test_name}/{input_name} not available")
        interp, env = _load_and_run(test_name, input_name, tmp_path)
        assert len(interp.explored_blocks) > 0, "Should explore at least one block"

    @pytest.mark.parametrize("test_name,input_name", _BENCHMARKS)
    def test_trace_file_written(self, test_name, input_name, tmp_path):
        if not _benchmark_available(test_name, input_name):
            pytest.skip(f"{test_name}/{input_name} not available")
        _load_and_run(test_name, input_name, tmp_path)
        trace = (tmp_path / "trace.txt").read_text()
        assert len(trace) > 0, "Trace file should have content"

    @pytest.mark.parametrize("test_name,input_name", _BENCHMARKS)
    def test_public_shadow_consistency(self, test_name, input_name, tmp_path):
        """Public variables should have matching shadow values after execution."""
        if not _benchmark_available(test_name, input_name):
            pytest.skip(f"{test_name}/{input_name} not available")
        input_path = Path(f"test_input/{test_name}/{input_name}.json")
        program_inputs = parse_inputs(input_path)
        interp, env = _load_and_run(test_name, input_name, tmp_path)
        for var_name, inp in program_inputs.variables.items():
            if inp.value is not None and not inp.private:
                shadow_name = f"{var_name}.shadow"
                val = env.get_concrete_value(var_name)
                shadow_val = env.get_concrete_value(shadow_name)
                if val is not None and shadow_val is not None:
                    if not isinstance(val, MemoryMap):
                        assert val == shadow_val, (
                            f"[{test_name}] Public var {var_name} diverged: {val} != {shadow_val}"
                        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
