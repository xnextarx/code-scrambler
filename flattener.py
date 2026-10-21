#!/usr/bin/env python3
"""
flattener.py — Converts C AST into flat basic blocks connected by goto statements.

This is the core engine of the C code scrambler.  It walks a parsed C AST and
converts every function body into a list of BasicBlocks where each block ends
with an explicit control-flow statement (goto, conditional goto, or return).
The blocks can then be shuffled freely while the goto chain preserves the
original execution order.

Supported C constructs:
    • Sequential statements       • if / else / else-if chains
    • while loops                  • for loops (including C99 init-decls)
    • do-while loops               • switch / case / default (with fall-through)
    • break / continue             • goto / labels (user-defined)
    • Variable declarations        • return
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pycparser import c_ast, c_generator


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BasicBlock:
    """A basic block: a label followed by statements ending in a control-flow jump."""
    label: str
    statements: List[str] = field(default_factory=list)


@dataclass
class _BreakContinueCtx:
    """Tracks the goto targets for break / continue inside loops and switches."""
    break_label: str
    continue_label: Optional[str] = None   # None → inside a switch (continue N/A)


# ═══════════════════════════════════════════════════════════════════════════
# Label generator
# ═══════════════════════════════════════════════════════════════════════════

class _LabelGen:
    """Generates unique labels (__sc_L0, __sc_L1, …) and temp vars."""

    def __init__(self) -> None:
        self._n = 0

    def label(self) -> str:
        lbl = f"__sc_L{self._n}"
        self._n += 1
        return lbl

    def temp(self) -> str:
        name = f"__sc_t{self._n}"
        self._n += 1
        return name


# ═══════════════════════════════════════════════════════════════════════════
# Flattener
# ═══════════════════════════════════════════════════════════════════════════

class Flattener:
    """
    Converts one C function body into a list of :class:`BasicBlock` instances.

    Usage::

        fl = Flattener()
        entry, blocks, hoisted = fl.flatten_function(func_def_ast_node)
    """

    def __init__(self) -> None:
        self._gen    = _LabelGen()
        self._cgen   = c_generator.CGenerator()
        self._blocks: List[BasicBlock]         = []
        self._hoisted: List[str]               = []
        self._ctx: List[_BreakContinueCtx]     = []

    # ── helpers ──────────────────────────────────────────────────────────

    def _lbl(self) -> str:
        return self._gen.label()

    def _tmp(self) -> str:
        return self._gen.temp()

    def _blk(self, label: str) -> BasicBlock:
        b = BasicBlock(label=label)
        self._blocks.append(b)
        return b

    def _c(self, node) -> str:
        """AST node → C-code string (stripped of surrounding whitespace)."""
        return self._cgen.visit(node).strip() if node else ""

    def _c_stmt(self, node) -> str:
        """AST node → C statement string (trailing `;` guaranteed)."""
        s = self._c(node)
        return s if s.endswith(";") else s + ";"

    # ── declaration helpers ──────────────────────────────────────────────

    def _decl_no_init(self, node: c_ast.Decl) -> str:
        """Render *Decl* without its initializer (for hoisting)."""
        saved_init = node.init
        node.init = None

        # Fix unsized arrays:  int a[] = {…}  →  int a[N]
        dim_fixed = False
        if (isinstance(node.type, c_ast.ArrayDecl)
                and node.type.dim is None
                and saved_init
                and isinstance(saved_init, c_ast.InitList)):
            node.type.dim = c_ast.Constant("int", str(len(saved_init.exprs)))
            dim_fixed = True

        text = self._cgen.visit(node).strip().rstrip(";").strip()

        # Restore the AST so we don't corrupt it for later use
        node.init = saved_init
        if dim_fixed:
            node.type.dim = None

        # Strip const / volatile so the later assignment is legal
        text = text.replace("const ", "").replace("volatile ", "")
        return text

    def _can_split_decl(self, node: c_ast.Decl) -> bool:
        """Can we hoist the decl and emit the init as a separate assignment?"""
        full = self._c(node)
        if "static" in full:
            return False                       # init must execute only once
        if not node.init:
            return True                        # nothing to split
        # char s[] = "hello"  →  can't assign string to array
        if isinstance(node.type, c_ast.ArrayDecl) and isinstance(node.init, c_ast.Constant):
            return False
        # Struct / union with {…}  →  would need field-by-field
        if isinstance(node.init, c_ast.InitList) and not isinstance(node.type, c_ast.ArrayDecl):
            return False
        # Multi-dim array (nested InitList)
        if isinstance(node.init, c_ast.InitList):
            if any(isinstance(e, c_ast.InitList) for e in node.init.exprs):
                return False
        return True

    # ── main entry point ─────────────────────────────────────────────────

    def flatten_function(
        self, fdef: c_ast.FuncDef
    ) -> Tuple[str, List[BasicBlock], List[str]]:
        """
        Flatten one function.

        Returns
        -------
        entry_label : str
            Label of the first block to execute.
        blocks : list[BasicBlock]
            All blocks (in arbitrary order — the caller will shuffle them).
        hoisted_decls : list[str]
            Variable declarations that must appear at the top of the function.
        """
        self._blocks  = []
        self._hoisted = []
        self._ctx     = []

        end   = self._lbl()
        entry = self._flatten(fdef.body, end)

        # Implicit return at end-of-function
        eb = self._blk(end)
        eb.statements.append("return;" if self._is_void(fdef) else "return 0;")

        # Hoisted decls were collected in reverse order (last-to-first
        # traversal), so reverse them back to source order.
        return entry, self._blocks, list(reversed(self._hoisted))

    @staticmethod
    def _is_void(fdef: c_ast.FuncDef) -> bool:
        try:
            ft = fdef.decl.type
            if isinstance(ft, c_ast.FuncDecl):
                rt = ft.type
                if isinstance(rt, c_ast.TypeDecl) and isinstance(rt.type, c_ast.IdentifierType):
                    return "void" in rt.type.names
        except Exception:                                       # pragma: no cover
            pass
        return False

    # ── recursive dispatcher ─────────────────────────────────────────────

    def _flatten(self, node, nxt: str) -> str:
        """Flatten *node* and return its entry label."""
        if node is None:
            return nxt

        t = type(node)
        if t is c_ast.Compound:       return self._do_compound(node, nxt)
        if t is c_ast.If:             return self._do_if(node, nxt)
        if t is c_ast.While:          return self._do_while(node, nxt)
        if t is c_ast.For:            return self._do_for(node, nxt)
        if t is c_ast.DoWhile:        return self._do_dowhile(node, nxt)
        if t is c_ast.Switch:         return self._do_switch(node, nxt)
        if t is c_ast.Return:         return self._do_return(node, nxt)
        if t is c_ast.Break:          return self._do_break(node, nxt)
        if t is c_ast.Continue:       return self._do_continue(node, nxt)
        if t is c_ast.Goto:           return self._do_goto(node, nxt)
        if t is c_ast.Label:          return self._do_label(node, nxt)
        if t is c_ast.Decl:           return self._do_decl(node, nxt)
        if t is c_ast.EmptyStatement: return nxt
        return self._do_expr(node, nxt)

    # ── Compound ─────────────────────────────────────────────────────────

    def _do_compound(self, node: c_ast.Compound, nxt: str) -> str:
        if not node.block_items:
            return nxt
        cur = nxt
        for stmt in reversed(node.block_items):
            cur = self._flatten(stmt, cur)
        return cur

    # ── If / Else ────────────────────────────────────────────────────────

    def _do_if(self, node: c_ast.If, nxt: str) -> str:
        t_entry = self._flatten(node.iftrue, nxt)
        f_entry = self._flatten(node.iffalse, nxt) if node.iffalse else nxt

        lbl = self._lbl()
        blk = self._blk(lbl)
        blk.statements.append(f"if ({self._c(node.cond)}) goto {t_entry};")
        blk.statements.append(f"goto {f_entry};")
        return lbl

    # ── While ────────────────────────────────────────────────────────────

    def _do_while(self, node: c_ast.While, nxt: str) -> str:
        cond = self._lbl()
        self._ctx.append(_BreakContinueCtx(break_label=nxt, continue_label=cond))

        body_entry = self._flatten(node.stmt, cond)

        self._ctx.pop()

        cb = self._blk(cond)
        cb.statements.append(f"if ({self._c(node.cond)}) goto {body_entry};")
        cb.statements.append(f"goto {nxt};")
        return cond

    # ── For ──────────────────────────────────────────────────────────────

    def _do_for(self, node: c_ast.For, nxt: str) -> str:
        cond_lbl = self._lbl()
        upd_lbl  = self._lbl()

        self._ctx.append(_BreakContinueCtx(break_label=nxt, continue_label=upd_lbl))
        body_entry = self._flatten(node.stmt, upd_lbl)
        self._ctx.pop()

        # — update block —
        ub = self._blk(upd_lbl)
        if node.next:
            ub.statements.append(self._c(node.next) + ";")
        ub.statements.append(f"goto {cond_lbl};")

        # — condition block —
        cb = self._blk(cond_lbl)
        if node.cond:
            cb.statements.append(f"if ({self._c(node.cond)}) goto {body_entry};")
            cb.statements.append(f"goto {nxt};")
        else:
            cb.statements.append(f"goto {body_entry};")      # for(;;)

        # — init —
        if node.init:
            il = self._lbl()
            ib = self._blk(il)
            if isinstance(node.init, c_ast.DeclList):
                for d in node.init.decls:
                    self._hoisted.append(self._decl_no_init(d) + ";")
                    if d.init:
                        ib.statements.append(f"{d.name} = {self._c(d.init)};")
            else:
                ib.statements.append(self._c(node.init) + ";")
            ib.statements.append(f"goto {cond_lbl};")
            return il

        return cond_lbl

    # ── Do-While ─────────────────────────────────────────────────────────

    def _do_dowhile(self, node: c_ast.DoWhile, nxt: str) -> str:
        cond_lbl = self._lbl()

        self._ctx.append(_BreakContinueCtx(break_label=nxt, continue_label=cond_lbl))
        body_entry = self._flatten(node.stmt, cond_lbl)
        self._ctx.pop()

        cb = self._blk(cond_lbl)
        cb.statements.append(f"if ({self._c(node.cond)}) goto {body_entry};")
        cb.statements.append(f"goto {nxt};")
        return body_entry                                      # body runs first

    # ── Switch ───────────────────────────────────────────────────────────

    def _do_switch(self, node: c_ast.Switch, nxt: str) -> str:
        self._ctx.append(_BreakContinueCtx(break_label=nxt))  # no continue in switch

        groups = self._group_cases(node.stmt)
        if not groups:
            self._ctx.pop()
            return nxt

        # Flatten case bodies (last → first for fall-through)
        info = []                              # [(case_node, entry_label), …]
        fall = nxt
        for case_node, stmts in reversed(groups):
            entry = fall
            for s in reversed(stmts):
                entry = self._flatten(s, entry)
            info.insert(0, (case_node, entry))
            fall = entry

        self._ctx.pop()

        # Dispatch block:  evaluate once, then chain of  if (== val) goto …
        dl  = self._lbl()
        db  = self._blk(dl)
        tmp = self._tmp()
        self._hoisted.append(f"long long {tmp};")
        db.statements.append(f"{tmp} = (long long)({self._c(node.cond)});")

        default_target = nxt
        for cn, entry in info:
            if isinstance(cn, c_ast.Case):
                db.statements.append(
                    f"if ({tmp} == (long long)({self._c(cn.expr)})) goto {entry};"
                )
            elif isinstance(cn, c_ast.Default):
                default_target = entry

        db.statements.append(f"goto {default_target};")
        return dl

    @staticmethod
    def _group_cases(compound) -> list:
        """Group a switch body into ``(case_node, [body_stmts])`` pairs."""
        if not isinstance(compound, c_ast.Compound) or not compound.block_items:
            return []

        groups: list = []
        cur_case  = None
        cur_stmts: list = []

        def flush():
            nonlocal cur_case, cur_stmts
            if cur_case is not None or cur_stmts:
                groups.append((cur_case, cur_stmts))
                cur_case  = None
                cur_stmts = []

        def walk(items):
            nonlocal cur_case, cur_stmts
            for item in items:
                if isinstance(item, (c_ast.Case, c_ast.Default)):
                    flush()
                    cur_case = item
                    if item.stmts:
                        walk(item.stmts)       # handles  case 1: case 2: stmt;
                else:
                    cur_stmts.append(item)

        walk(compound.block_items)
        flush()
        return groups

    # ── Terminal statements ──────────────────────────────────────────────

    def _do_return(self, node: c_ast.Return, _nxt: str) -> str:
        lbl = self._lbl()
        blk = self._blk(lbl)
        blk.statements.append(self._c_stmt(node))
        return lbl

    def _do_break(self, _node, _nxt: str) -> str:
        lbl = self._lbl()
        blk = self._blk(lbl)
        if self._ctx:
            blk.statements.append(f"goto {self._ctx[-1].break_label};")
        return lbl

    def _do_continue(self, _node, _nxt: str) -> str:
        lbl = self._lbl()
        blk = self._blk(lbl)
        for ctx in reversed(self._ctx):
            if ctx.continue_label is not None:
                blk.statements.append(f"goto {ctx.continue_label};")
                break
        return lbl

    def _do_goto(self, node: c_ast.Goto, _nxt: str) -> str:
        lbl = self._lbl()
        blk = self._blk(lbl)
        blk.statements.append(f"goto {node.name};")
        return lbl

    def _do_label(self, node: c_ast.Label, nxt: str) -> str:
        stmt_entry = self._flatten(node.stmt, nxt)
        blk = self._blk(node.name)             # keeps original label name
        blk.statements.append(f"goto {stmt_entry};")
        return node.name

    # ── Declarations ─────────────────────────────────────────────────────

    def _do_decl(self, node: c_ast.Decl, nxt: str) -> str:
        # Function declarations → skip
        if isinstance(node.type, c_ast.FuncDecl):
            return nxt

        # If we can't safely split → hoist the whole thing (with init)
        if not self._can_split_decl(node):
            full = self._c(node).rstrip().rstrip(";").strip()
            self._hoisted.append(full + ";")
            return nxt

        # Hoist declaration (without initializer)
        self._hoisted.append(self._decl_no_init(node) + ";")

        if not node.init:
            return nxt

        # Create an initialisation block
        lbl = self._lbl()
        blk = self._blk(lbl)

        if isinstance(node.type, c_ast.ArrayDecl) and isinstance(node.init, c_ast.InitList):
            for i, expr in enumerate(node.init.exprs):
                blk.statements.append(f"{node.name}[{i}] = {self._c(expr)};")
        else:
            blk.statements.append(f"{node.name} = {self._c(node.init)};")

        blk.statements.append(f"goto {nxt};")
        return lbl

    # ── Expressions (fall-through) ───────────────────────────────────────

    def _do_expr(self, node, nxt: str) -> str:
        lbl = self._lbl()
        blk = self._blk(lbl)
        blk.statements.append(self._c(node) + ";")
        blk.statements.append(f"goto {nxt};")
        return lbl
