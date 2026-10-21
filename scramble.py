#!/usr/bin/env python3
"""
scramble.py — C Code Scrambler
===============================

Reorders every instruction in a C source file while preserving the original
execution flow.  The tool:

  1. Parses the C source into an AST  (via *pycparser*).
  2. Converts each function body into a flat list of *basic blocks*,
     where every block ends with a ``goto``, conditional ``goto``, or
     ``return``.
  3. Randomly shuffles the blocks.
  4. Emits the shuffled blocks — the ``goto`` chain guarantees the
     original execution order.

Usage
-----
::

    python scramble.py input.c                     # stdout
    python scramble.py input.c -o out.c            # write to file
    python scramble.py input.c -o out.c --verify   # compile & compare outputs
    python scramble.py input.c --seed 42           # reproducible shuffle
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile

try:
    from pycparser import c_parser, c_ast, c_generator
    CGenerator = c_generator.CGenerator

except ImportError:
    print(
        "Error: pycparser is required.\n"
        "       Install it with:  pip install pycparser",
        file=sys.stderr,
    )
    sys.exit(1)

from flattener import Flattener


# ═══════════════════════════════════════════════════════════════════════════
# Fake type definitions (so pycparser can parse without a real preprocessor)
# ═══════════════════════════════════════════════════════════════════════════

FAKE_TYPEDEFS = """\
typedef int            size_t;
typedef int            ssize_t;
typedef int            ptrdiff_t;
typedef int            wchar_t;
typedef unsigned int   mode_t;
typedef unsigned int   pid_t;
typedef unsigned int   uid_t;
typedef unsigned int   gid_t;
typedef long           off_t;
typedef long           time_t;
typedef long           clock_t;
typedef void           FILE;
typedef int            bool;
typedef int            _Bool;
typedef unsigned int   uint32_t;
typedef unsigned long  uint64_t;
typedef int            int32_t;
typedef long           int64_t;
typedef unsigned char  uint8_t;
typedef char           int8_t;
typedef unsigned short uint16_t;
typedef short          int16_t;
typedef void*          va_list;
"""

_FAKE_NAMES: set[str] = {
    "size_t", "ssize_t", "ptrdiff_t", "wchar_t",
    "mode_t", "pid_t", "uid_t", "gid_t", "off_t",
    "time_t", "clock_t", "FILE", "bool", "_Bool",
    "uint32_t", "uint64_t", "int32_t", "int64_t",
    "uint8_t", "int8_t", "uint16_t", "int16_t",
    "va_list",
}


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def _extract_preprocessor(source: str) -> tuple[str, list[str]]:
    """Split preprocessor directives from C code.

    Returns ``(code_without_directives, list_of_directives)``.
    """
    directives: list[str] = []
    code_lines: list[str] = []

    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            directives.append(line)
            code_lines.append("")          # keep line numbers sane
        else:
            code_lines.append(line)

    return "\n".join(code_lines), directives


# ═══════════════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse(code: str) -> c_ast.FileAST:
    """Parse C code into a *pycparser* AST (with fake-typedef fallback)."""
    parser = c_parser.CParser()

    # 1) Try raw
    try:
        return parser.parse(code, filename="<input>")
    except c_parser.ParseError:
        pass

    # 2) Prepend fake typedefs
    try:
        return parser.parse(FAKE_TYPEDEFS + code, filename="<input>")
    except c_parser.ParseError as exc:
        print(f"Parse error: {exc}", file=sys.stderr)
        print(
            "\nHint: make sure the input is valid C.  Preprocessor directives\n"
            "      (#include, #define …) are stripped automatically, but complex\n"
            "      macros or non-standard extensions may cause failures.",
            file=sys.stderr,
        )
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Code generation
# ═══════════════════════════════════════════════════════════════════════════

def _func_signature(fdef: c_ast.FuncDef, cgen: CGenerator) -> str:
    """Return the function signature (e.g. ``int main(int argc, char **argv)``)."""
    return cgen.visit(fdef.decl).strip()


def _emit_function(
    signature: str,
    entry_label: str,
    blocks: list,
    hoisted: list[str],
) -> str:
    """Render one scrambled function as C source."""
    lines: list[str] = [signature, "{"]

    # ── hoisted declarations (deduplicated, preserving order) ──
    seen: set[str] = set()
    unique: list[str] = []
    for d in hoisted:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    if unique:
        lines.append("    /* ── hoisted declarations ── */")
        for d in unique:
            lines.append(f"    {d}")
        lines.append("")

    # ── jump to the original entry point ──
    lines.append(f"    goto {entry_label};")
    lines.append("")

    # ── scrambled blocks ──
    for blk in blocks:
        lines.append(f"  {blk.label}:")
        for stmt in blk.statements:
            lines.append(f"    {stmt}")
        lines.append("")

    lines.append("}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main scramble pipeline
# ═══════════════════════════════════════════════════════════════════════════

def scramble(source: str, *, seed: int | None = None) -> str:
    """Scramble a complete C translation unit.

    Parameters
    ----------
    source : str
        Full C source code.
    seed : int or None
        Random seed for reproducible output.

    Returns
    -------
    str
        The scrambled C source.
    """
    if seed is not None:
        random.seed(seed)

    code, directives = _extract_preprocessor(source)
    ast = _parse(code)
    cgen = c_generator.CGenerator()

    parts: list[str] = []

    # ── preserved preprocessor directives ──
    if directives:
        parts.extend(directives)
        parts.append("")

    # ── process every top-level declaration ──
    for ext in ast.ext:
        # Skip our own fake typedefs
        if isinstance(ext, c_ast.Typedef) and ext.name in _FAKE_NAMES:
            continue

        if isinstance(ext, c_ast.FuncDef):
            sig = _func_signature(ext, cgen)
            fl  = Flattener()
            entry, blocks, hoisted = fl.flatten_function(ext)

            random.shuffle(blocks)

            parts.append(_emit_function(sig, entry, blocks, hoisted))
            parts.append("")
        else:
            # Global variable, typedef, struct, etc. → keep as-is
            gen_text = cgen.visit(ext).strip()
            if not gen_text.endswith(";"):
                gen_text += ";"
            parts.append(gen_text)
            parts.append("")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Verification (optional)
# ═══════════════════════════════════════════════════════════════════════════

def _verify(original_path: str, scrambled_code: str) -> bool:
    """Compile & run both versions and compare stdout + exit code."""
    scr_fd, scr_path = tempfile.mkstemp(suffix=".c")
    orig_bin = tempfile.mktemp()
    scr_bin  = tempfile.mktemp()

    try:
        with os.fdopen(scr_fd, "w") as f:
            f.write(scrambled_code)

        # Compile original
        r1 = subprocess.run(
            ["gcc", "-std=c99", "-w", "-o", orig_bin, original_path],
            capture_output=True, text=True,
        )
        if r1.returncode != 0:
            print(f"⚠  Original failed to compile:\n{r1.stderr}", file=sys.stderr)
            return False

        # Compile scrambled
        r2 = subprocess.run(
            ["gcc", "-std=c99", "-w", "-o", scr_bin, scr_path],
            capture_output=True, text=True,
        )
        if r2.returncode != 0:
            print(f"✗  Scrambled code failed to compile:\n{r2.stderr}", file=sys.stderr)
            return False

        # Run both
        r3 = subprocess.run([orig_bin], capture_output=True, text=True, timeout=10)
        r4 = subprocess.run([scr_bin],  capture_output=True, text=True, timeout=10)

        ok = r3.stdout == r4.stdout and r3.returncode == r4.returncode

        if ok:
            print("✓  Verification PASSED — outputs match.", file=sys.stderr)
        else:
            print("✗  Verification FAILED — outputs differ!", file=sys.stderr)
            if r3.stdout != r4.stdout:
                print(f"   Original stdout : {r3.stdout[:300]!r}", file=sys.stderr)
                print(f"   Scrambled stdout: {r4.stdout[:300]!r}", file=sys.stderr)
            if r3.returncode != r4.returncode:
                print(f"   Original rc={r3.returncode}, Scrambled rc={r4.returncode}",
                      file=sys.stderr)
        return ok

    except FileNotFoundError:
        print("⚠  gcc not found — skipping verification.", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("⚠  Execution timed out (10 s).", file=sys.stderr)
        return False
    finally:
        for p in (scr_path, orig_bin, scr_bin):
            try:
                os.unlink(p)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="C Code Scrambler — reorder every instruction while "
                    "preserving execution flow.",
    )
    ap.add_argument("input", help="Input C source file (use '-' for stdin)")
    ap.add_argument("-o", "--output", help="Output file (default: stdout)")
    ap.add_argument("-s", "--seed", type=int, default=None,
                    help="Random seed for reproducible results")
    ap.add_argument("-v", "--verify", action="store_true",
                    help="Compile & run both versions and compare (requires gcc)")

    args = ap.parse_args()

    # ── read input ──
    if args.input == "-":
        source     = sys.stdin.read()
        input_path = None
    else:
        if not os.path.isfile(args.input):
            print(f"Error: file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        with open(args.input) as f:
            source = f.read()
        input_path = os.path.abspath(args.input)

    # ── scramble ──
    result = scramble(source, seed=args.seed)

    # ── write output ──
    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"Scrambled → {args.output}", file=sys.stderr)
    else:
        print(result)

    # ── optional verification ──
    if args.verify:
        if input_path is None:
            print("⚠  --verify requires a file input (not stdin).", file=sys.stderr)
        else:
            _verify(input_path, result)


if __name__ == "__main__":
    main()
