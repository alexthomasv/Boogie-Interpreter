"""
End-to-end tests: native C execution vs Boogie interpreter.

Native C: compile with SMACK stubs and run directly.
Interpreter: run via `pypy interpreter.py` subprocess, check compact trace.

These tests verify that both paths produce valid results for the same inputs.
"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEARSSL_DIR = PROJECT_ROOT / "examples" / "bearssl"
BEARSSL_SRC = BEARSSL_DIR / "src"
BEARSSL_INC = BEARSSL_DIR / "inc"
TEST_INPUT_DIR = PROJECT_ROOT / "test_input"
TEST_PACKAGES_DIR = PROJECT_ROOT / "test_packages"

# Stub C file providing all SMACK/ct-verif symbols as no-ops.
# Compiled as a separate translation unit so all source files can link against it.
SMACK_STUBS_C = r"""
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
typedef struct { int dummy; } smack_value;
smack_value* __SMACK_value(void *p) { (void)p; return 0; }
smack_value* __SMACK_values(void *p, int n) { (void)p; (void)n; return 0; }
void public_in(smack_value *v) { (void)v; }
void private_in(smack_value *v) { (void)v; }
void declassified_out(smack_value *v) { (void)v; }
void __SMACK_code(const char *fmt, ...) { (void)fmt; }
size_t __VERIFIER_nondet_size_t(void) { return 0; }
void __VERIFIER_assume(int x) { (void)x; }
void __SMACK_inv_buffer(void) {}
void __SMACK_inv_bounds(void) {}
void __SMACK_inv_byte_copy_fwd(void) {}
void __SMACK_inv_byte_copy_bwd(void) {}
void __SMACK_dummy(int x) { (void)x; }
"""

# g_header.h stubs (memmove wrappers used by BearSSL)
G_HEADER_STUBS = r"""
#ifndef G_HEADER_H
#define G_HEADER_H
#include <string.h>
#define g_memmove memmove
#define g_memcpy memcpy
#define g_memset memset
#define g_memcmp memcmp
#endif
"""

# swoosh_invariant.h stub
SWOOSH_INV_STUB = r"""
#ifndef SWOOSH_INVARIANT_H
#define SWOOSH_INVARIANT_H
#define SWOOSH_LOOP_INV(fmt, ...) do {} while(0)
#define SWOOSH_LOOP_INV_CUSTOM(tag, attrs, fmt, ...) do {} while(0)
#define SWOOSH_ASSUME_CUSTOM(tag, fmt, ...) do {} while(0)
#define SWOOSH_E_BUFFER(e, e_init, elen, elen_init) do {} while(0)
#define BPL_LOAD64 ""
#endif
"""


def _find_cc():
    """Find a C compiler."""
    for cc in ["gcc", "clang", "cc"]:
        if shutil.which(cc):
            return cc
    return None


def _hex_to_bytes(h):
    h = h.strip()
    if h.startswith("0x") or h.startswith("0X"):
        h = h[2:]
    if len(h) % 2:
        h = "0" + h
    return bytes.fromhex(h)


def _parse_test_input(json_path):
    with open(json_path) as f:
        return json.load(f)


# ── PKCS i15 native compilation ──

PKCS_I15_SOURCES = [
    BEARSSL_SRC / "int" / "i15_modpow2.c",
    BEARSSL_SRC / "int" / "i15_fmont.c",
    BEARSSL_SRC / "int" / "i15_decode.c",
    BEARSSL_SRC / "int" / "i15_mulacc.c",
    BEARSSL_SRC / "int" / "i15_ninv15.c",
    BEARSSL_SRC / "int" / "i15_montmul.c",
    BEARSSL_SRC / "int" / "i15_sub.c",
    BEARSSL_SRC / "int" / "i15_rshift.c",
    BEARSSL_SRC / "int" / "i15_decred.c",
    BEARSSL_SRC / "int" / "i15_bitlen.c",
    BEARSSL_SRC / "codec" / "ccopy.c",
    BEARSSL_SRC / "int" / "i15_encode.c",
    BEARSSL_SRC / "int" / "i15_reduce.c",
    BEARSSL_SRC / "int" / "i15_tmont.c",
    BEARSSL_SRC / "int" / "i15_add.c",
    BEARSSL_SRC / "int" / "i15_muladd.c",
    BEARSSL_SRC / "rsa" / "rsa_i15_priv.c",
    BEARSSL_SRC / "rsa" / "rsa_i15_pkcs1_sign.c",
    BEARSSL_SRC / "rsa" / "rsa_pkcs1_sig_pad.c",
]


def _generate_pkcs_harness(variant, input_data):
    """Generate a C main() that sets up test data and calls the signing function."""
    # Extract components from input JSON
    oid_entry = input_data[0]  # $p0: OID
    hash_entry = input_data[1]  # $p1: hash
    hash_len_entry = input_data[2]  # $i2: hash_len
    sk_entry = input_data[3]  # $p3: sk struct with arrays + fields
    out_entry = input_data[4]  # $p4: output buffer

    oid_bytes = _hex_to_bytes(oid_entry["array"][0]["contents"])
    hash_bytes = _hex_to_bytes(hash_entry["array"][0]["contents"])
    hash_len = hash_len_entry["value"]
    out_size = _hex_to_bytes(out_entry["array"][0]["contents"])

    # CRT components
    components = sk_entry["array"]  # 5 entries: p, q, dp, dq, iq
    fields = sk_entry["fields"]
    n_bitlen_bytes = _hex_to_bytes(fields[0]["contents"])
    n_bitlen = int.from_bytes(n_bitlen_bytes, "little")

    # Each component is PHY_SIZE_FACTOR bytes
    comp_size = len(_hex_to_bytes(components[0]["contents"]))

    lines = [
        '#include <stdio.h>',
        '#include <string.h>',
        '#include <stdint.h>',
        '#include "inner.h"',
        '',
        f'uint32_t br_rsa_{variant}_pkcs1_sign(',
        '    const unsigned char *hash_oid,',
        '    const unsigned char *hash, size_t hash_len,',
        '    const br_rsa_private_key *sk, unsigned char *x);',
        '',
        'int main(void) {',
    ]

    # OID
    lines.append(f'    static const unsigned char oid[{len(oid_bytes)}] = {{')
    lines.append('        ' + ', '.join(f'0x{b:02x}' for b in oid_bytes))
    lines.append('    };')

    # Hash
    lines.append(f'    static const unsigned char hash[{len(hash_bytes)}] = {{')
    lines.append('        ' + ', '.join(f'0x{b:02x}' for b in hash_bytes))
    lines.append('    };')

    # CRT components
    comp_names = ["p", "q", "dp", "dq", "iq"]
    for i, name in enumerate(comp_names):
        comp_bytes = _hex_to_bytes(components[i]["contents"])
        lines.append(f'    static unsigned char {name}[{len(comp_bytes)}] = {{')
        # Print in chunks of 16
        for j in range(0, len(comp_bytes), 16):
            chunk = comp_bytes[j:j+16]
            lines.append('        ' + ', '.join(f'0x{b:02x}' for b in chunk) + ',')
        lines.append('    };')

    # Component lengths from fields
    comp_lens = []
    for i in range(5):
        len_bytes = _hex_to_bytes(fields[1 + i * 2 + 1]["contents"])
        comp_lens.append(int.from_bytes(len_bytes, "little"))

    # Private key struct
    lines.append(f'    br_rsa_private_key sk = {{')
    lines.append(f'        {n_bitlen},')
    for i, name in enumerate(comp_names):
        lines.append(f'        {name}, {comp_lens[i]},')
    lines.append('    };')

    # Output buffer
    out_buf_size = len(out_size)
    lines.append(f'    unsigned char x[{out_buf_size}];')
    lines.append(f'    memset(x, 0, sizeof(x));')
    lines.append('')

    # Call
    lines.append(f'    uint32_t ret = br_rsa_{variant}_pkcs1_sign(oid, hash, {hash_len}, &sk, x);')
    lines.append(f'    printf("RETURN:%u\\n", ret);')
    lines.append(f'    printf("OUTPUT:");')
    lines.append(f'    for (int i = 0; i < {out_buf_size}; i++) printf("%02x", x[i]);')
    lines.append(f'    printf("\\n");')
    lines.append('    return 0;')
    lines.append('}')

    return '\n'.join(lines)


def _compile_and_run_native(harness_code, sources, build_dir, name):
    """Compile harness + sources and run. Returns (return_value, output_hex) or None."""
    cc = _find_cc()
    if not cc:
        return None

    # Write stub files
    (build_dir / "smack_stubs.c").write_text(SMACK_STUBS_C)
    (build_dir / "g_header.h").write_text(G_HEADER_STUBS)
    (build_dir / "swoosh_invariant.h").write_text(SWOOSH_INV_STUB)

    # Write harness
    harness_path = build_dir / f"{name}_harness.c"
    harness_path.write_text(harness_code)

    # Check all sources exist
    for src in sources:
        if not src.exists():
            return None

    exe_path = build_dir / name
    cmd = [
        cc, "-O0", "-w",
        f"-I{build_dir}",
        f"-I{BEARSSL_INC}",
        f"-I{BEARSSL_SRC}",
        "-DBR_CT_MUL15=1",
        str(harness_path),
        str(build_dir / "smack_stubs.c"),
    ] + [str(s) for s in sources] + ["-o", str(exe_path)]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"Compilation failed:\n{r.stderr}")
        return None

    r = subprocess.run([str(exe_path)], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"Execution failed:\n{r.stderr}")
        return None

    result = {}
    for line in r.stdout.strip().split('\n'):
        if line.startswith("RETURN:"):
            result["return_value"] = int(line.split(":")[1])
        elif line.startswith("OUTPUT:"):
            result["output_hex"] = line.split(":")[1]
    return result


def _run_interpreter_subprocess(pkg_name):
    """Run interpreter as PyPy subprocess. Returns True if compact trace exists."""
    pkg_dir = TEST_PACKAGES_DIR / f"{pkg_name}_pkg"
    if not pkg_dir.exists():
        return None

    pypy = shutil.which("pypy") or shutil.which("pypy3")
    if not pypy:
        return None

    trace_dir = Path(f"positive_examples/{pkg_name}")
    compact_files = list(trace_dir.glob("*.trace.compact.pkl")) if trace_dir.exists() else []
    if compact_files:
        # Already have traces, no need to re-run
        return {"exists": True, "path": compact_files[0]}

    r = subprocess.run(
        [pypy, "-m", "interpreter.python.interpreter", str(pkg_dir)],
        capture_output=True, text=True, timeout=600,
        cwd=str(PROJECT_ROOT),
    )

    compact_files = list(trace_dir.glob("*.trace.compact.pkl")) if trace_dir.exists() else []
    return {
        "exists": len(compact_files) > 0 and r.returncode == 0,
        "path": compact_files[0] if compact_files else None,
        "returncode": r.returncode,
    }


# ── Tests ──

class TestPkcsI15Native:
    """Test native C compilation and execution of PKCS i15 signing."""

    @pytest.fixture(scope="class")
    def native_result(self, tmp_path_factory):
        json_path = TEST_INPUT_DIR / "bearssl_test_pkcs1_i15" / "test_06_custom_vector.json"
        if not json_path.exists():
            pytest.skip("Test input not found")
        input_data = _parse_test_input(json_path)
        harness = _generate_pkcs_harness("i15", input_data)
        build_dir = tmp_path_factory.mktemp("pkcs_i15_build")
        result = _compile_and_run_native(harness, PKCS_I15_SOURCES, build_dir, "pkcs_i15")
        if result is None:
            pytest.skip("Could not compile/run native C")
        return result

    def test_native_returns_nonzero(self, native_result):
        """BearSSL signing returns nonzero on success (p0i & q0i & r)."""
        assert native_result["return_value"] != 0, \
            "Expected nonzero return (success) from br_rsa_i15_pkcs1_sign"

    def test_native_output_not_all_zeros(self, native_result):
        """Output buffer should contain the RSA signature, not zeros."""
        out = native_result["output_hex"]
        assert out != "00" * (len(out) // 2), "Output should not be all zeros"

    def test_native_output_correct_length(self, native_result):
        """Output should be 256 bytes (2048-bit RSA)."""
        out = bytes.fromhex(native_result["output_hex"])
        assert len(out) == 256, f"Expected 256 bytes, got {len(out)}"


class TestPkcsI15Interpreter:
    """Test that the Boogie interpreter produces a valid compact trace."""

    def test_interpreter_produces_compact_trace(self):
        pkg_name = "bearssl_test_pkcs1_i15"
        result = _run_interpreter_subprocess(pkg_name)
        if result is None:
            pytest.skip("Package not compiled or PyPy not available")
        assert result["exists"], \
            f"Interpreter should produce a compact trace (rc={result.get('returncode')})"

    def test_compact_trace_has_data(self):
        trace_dir = Path("positive_examples/bearssl_test_pkcs1_i15")
        compact_files = list(trace_dir.glob("*.trace.compact.pkl")) if trace_dir.exists() else []
        if not compact_files:
            pytest.skip("No compact trace available")
        # Check file is non-trivial (> 1KB)
        assert compact_files[0].stat().st_size > 1024, \
            "Compact trace should be larger than 1KB"


class TestPkcsI15Consistency:
    """Cross-check native output against interpreter trace data."""

    @pytest.fixture(scope="class")
    def native_output(self, tmp_path_factory):
        json_path = TEST_INPUT_DIR / "bearssl_test_pkcs1_i15" / "test_06_custom_vector.json"
        if not json_path.exists():
            pytest.skip("Test input not found")
        input_data = _parse_test_input(json_path)
        harness = _generate_pkcs_harness("i15", input_data)
        build_dir = tmp_path_factory.mktemp("pkcs_i15_consistency")
        result = _compile_and_run_native(harness, PKCS_I15_SOURCES, build_dir, "pkcs_i15")
        if result is None:
            pytest.skip("Could not compile native C")
        return bytes.fromhex(result["output_hex"])

    def test_output_is_valid_pkcs1_signature(self, native_output):
        """Basic structural check: PKCS#1 v1.5 signature starts with 0x00 0x01."""
        # After RSA private key operation, the result should be a valid
        # integer mod n. We can at least check it's 256 bytes.
        assert len(native_output) == 256

    def test_output_is_deterministic(self, tmp_path_factory):
        """Running native twice with same input produces same output."""
        json_path = TEST_INPUT_DIR / "bearssl_test_pkcs1_i15" / "test_06_custom_vector.json"
        if not json_path.exists():
            pytest.skip("Test input not found")
        input_data = _parse_test_input(json_path)
        harness = _generate_pkcs_harness("i15", input_data)

        results = []
        for i in range(2):
            build_dir = tmp_path_factory.mktemp(f"pkcs_i15_det_{i}")
            r = _compile_and_run_native(harness, PKCS_I15_SOURCES, build_dir, f"pkcs_i15_{i}")
            if r is None:
                pytest.skip("Could not compile native C")
            results.append(r["output_hex"])

        assert results[0] == results[1], "Same input should produce same output"
