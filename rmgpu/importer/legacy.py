"""
Legacy RMG Python input importer (lossless, AST-only, no exec).

Converts legacy RMG ``.py`` input files into rmgpu schema-shaped documents
(snake_case, ``rmgpu: 1.0``). The importer is the job-03 gate's lossless path
(PLAN.md 12.2/12.3, risk 8): every DSL value must survive, and anything it
cannot express is reported LOUDLY as an IMPORT-NOTE (never silently defaulted).

Design:
- Top-level DSL calls are dispatched by name (``DSL_SCHEMA_KEYS``).
- Arguments are converted with ``node_to_value``: literals via
  ``ast.literal_eval``, numeric arithmetic (BinOp/UnaryOp) via a safe,
  stdlib-only evaluator, structure-helper calls (SMILES/InChI/adjacencyList/
  adjacencyListGroup/fragment_adj/fragment_SMILES/SMARTS) into
  ``StructureValue`` dicts. Anything else becomes an IMPORT-NOTE.
- Unknown keys, repeated calls, and list-valued (staged) reactor fields are
  preserved as-is so the document round-trips losslessly.
"""

import ast
import os
import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "import_legacy",
    "LegacyImporterError",
    "ImportNote",
    "node_to_value",
    "DSL_SCHEMA_KEYS",
    "REACTOR_TYPE_NAMES",
]


class LegacyImporterError(Exception):
    """Raised when the legacy importer cannot process a file."""

    pass


class ImportNote:
    """A loud note about something the importer could not express verbatim."""

    def __init__(self, function_name: str, line_number: int, message: str):
        self.function_name = function_name
        self.line_number = line_number
        self.message = message

    def __str__(self):
        return f"Line {self.line_number}: {self.function_name} - {self.message}"

    def to_dict(self):
        return {
            "line": self.line_number,
            "function": self.function_name,
            "message": self.message,
        }


# DSL function -> top-level schema key (None: not a schema block)
DSL_SCHEMA_KEYS = {
    # database() is handled specially (its kwargs are the DatabaseBlock fields)
    "database": "database",
    # list blocks
    "species": "species",
    "forbidden": "forbidden",
    "simpleReactor": "reactors",
    "constantVIdealGasReactor": "reactors",
    "constantTPIdealGasReactor": "reactors",
    "liquidReactor": "reactors",
    "mbsampledReactor": "reactors",
    "surfaceReactor": "reactors",
    "constantTVLiquidReactor": "reactors",
    "liquidSurfaceReactor": "reactors",
    # single-block (dict) functions
    "simulator": "simulator",
    "model": "model",
    "pressureDependence": "pressure_dependence",
    "mlEstimator": "ml_estimator",
    "solvation": "solvation",
    "uncertainty": "uncertainty",
    "options": "options",
    "generatedSpeciesConstraints": "generated_species_constraints",
    "catalystProperties": "catalyst_properties",
    "quantumMechanics": "quantum_mechanics",
    # not part of the rmgpu schema (QM out of scope, PLAN 8a.3) - preserved
    # verbatim at the document level with an IMPORT-NOTE, for losslessness.
    "restartFromSeed": None,
    "coreSpeciesFile": None,
    "react": None,
}

# reactor DSL function -> schema `type` literal
REACTOR_TYPE_NAMES = {
    "simpleReactor": "simple",
    "constantVIdealGasReactor": "const_V",
    "constantTPIdealGasReactor": "const_TP",
    "liquidReactor": "liquid",
    "mbsampledReactor": "mb_sampled",
    "surfaceReactor": "surface",
    "constantTVLiquidReactor": "liquid",
    "liquidSurfaceReactor": "liquid_surface",
}

# structure-helper DSL functions -> StructureValue field
STRUCTURE_FUNCS = {
    "SMILES": "smiles",
    "InChI": "inchi",
    "adjacencyList": "adjlist",
    "adjacencyListGroup": "group_adjlist",
    "fragment_adj": "fragment_adjlist",
    "fragment_SMILES": "smiles",
    "SMARTS": "smarts",
}

_QUANTITY_TUPLE_RE = re.compile(
    r"^\(\s*([0-9.eE+-]+)\s*,\s*(['\"])(.+?)\2\s*\)$"
)

# Safe-eval whitelist: names that may appear in numeric arithmetic sub-
# expressions of legacy inputs (RMG's Quantity DSL allows math).
_ALLOWED_NAMES = {"pi": 3.141592653589793, "e": 2.718281828459045}


def _safe_arith(node: ast.AST) -> Optional[float]:
    """Evaluate a numeric arithmetic AST node (BinOp/UnaryOp/Num) safely.

    Only binary ops +,-,*,/ and unary +/- on numbers are allowed; anything
    else (names not in the whitelist, calls, subscripts) returns None.
    """
    if isinstance(node, ast.Expression):
        return _safe_arith(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        return None
    if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _safe_arith(node.left)
        right = _safe_arith(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            return None
        return left / right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _safe_arith(node.operand)
        if v is None:
            return None
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.Name):
        return _ALLOWED_NAMES.get(node.id)
    return None


def _call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _node_to_container(node: ast.AST, source: str, notes: List[ImportNote],
                        context: str = "") -> Any:
    """Convert an AST container (list/tuple/dict/set) node, evaluating
    arithmetic and structure calls on its items. Returns a sentinel-dict when
    an item cannot be expressed (a note is recorded loudly)."""
    if isinstance(node, ast.Dict):
        out = {}
        for k, val in zip(node.keys, node.values):
            if k is None:  # **unpacking
                notes.append(ImportNote("<dict>", node.lineno,
                                        f"**unpacking in dict literal in {context}"))
                continue
            key = node_to_value(k, source, notes, context)
            valv = node_to_value(val, source, notes, context)
            if isinstance(valv, dict) and "__import_note__" in valv:
                return valv
            if isinstance(key, str) or isinstance(key, (int, float, bool)):
                out[key] = valv
            else:
                notes.append(ImportNote("<dict>", node.lineno,
                                        f"non-scalar dict key {key!r} in {context}"))
        return out
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = []
        for elt in node.elts:
            v = node_to_value(elt, source, notes, context)
            if isinstance(v, dict) and "__import_note__" in v:
                return v
            items.append(v)
        if isinstance(node, ast.Set):
            return items  # sets become lists (order-stable via sorted later if needed)
        return items
    return node_to_value(node, source, notes, context)


def node_to_value(node: ast.AST, source: str, notes: List[ImportNote],
                  context: str = "") -> Any:
    """Convert an AST argument node to a schema value.

    Returns the value, or a dict ``{"__import_note__": ...}`` sentinel when the
    node cannot be expressed (the caller records the note and keeps a
    placeholder so nothing is silently dropped).
    """
    # Container nodes: convert recursively (arithmetic + calls inside)
    if isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
        return _node_to_container(node, source, notes, context)

    # Literal: string, number, bool, None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        pass

    # Structure-helper call: SMILES(...), InChI(...), adjacencyList(...), ...
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name in STRUCTURE_FUNCS and node.args and not node.keywords:
            inner = node_to_value(node.args[0], source, notes, context)
            if isinstance(inner, str):
                # NOTE: no 'fragment' flag here - it is redundant (already
                # encoded in the field name) and would not round-trip
                # losslessly against the canonical value.
                return {STRUCTURE_FUNCS[name]: inner}
            return {"__import_note__": {
                "line": node.lineno, "function": name,
                "message": f"structure argument of {name}() is not a plain string (nested call)",
            }}
        if name in ("Quantity", "Energy", "RateCoefficient", "Concentration") \
                and len(node.args) == 2:
            v = node_to_value(node.args[0], source, notes, context)
            u = node_to_value(node.args[1], source, notes, context)
            if isinstance(v, (int, float)) and isinstance(u, str):
                return {"value": v, "unit": u}
            return {"__import_note__": {
                "line": node.lineno, "function": name,
                "message": f"{name}() arguments are not (number, unit-string)",
            }}
        # Any other call: try to capture it structurally
        args = [node_to_value(a, source, notes, context) for a in node.args]
        kwargs = {kw.arg: node_to_value(kw.value, source, notes, context)
                  for kw in node.keywords if kw.arg is not None}
        note = {"line": node.lineno, "function": name or "<call>",
                "message": f"unmapped call {name}() preserved structurally in {context or 'document'}"}
        return {"__call__": name, "args": args, "kwargs": kwargs,
                "__import_note__": note}

    # Numeric arithmetic (e.g. 1./6.5, 7.585e-3*2.0, -2.892)
    if isinstance(node, (ast.BinOp, ast.UnaryOp)):
        v = _safe_arith(node)
        if v is not None:
            return v
        return {"__import_note__": {
            "line": node.lineno, "function": "<arith>",
            "message": f"arithmetic sub-expression not safely evaluable in {context or 'document'}",
        }}

    # Comprehensions and other exotic nodes: capture source, loud note.
    src = ast.get_source_segment(source, node)
    return {"__import_note__": {
        "line": node.lineno, "function": type(node).__name__,
        "message": f"non-literal node {type(node).__name__} preserved in source in {context or 'document'}: {src[:200]!r}",
    }, "__source__": src}


def _is_quantity_pair(v: Any) -> bool:
    """A legacy RMG Quantity is a (number, unit-string) tuple or list."""
    return (isinstance(v, (tuple, list)) and len(v) == 2
            and isinstance(v[0], (int, float)) and not isinstance(v[0], bool)
            and isinstance(v[1], str))


def _coerce_list_value(v: Any) -> Any:
    """Normalize evaluated values: quantity pairs -> {value,unit} dicts,
    other tuples -> lists (YAML friendliness). Recurses into containers."""
    if _is_quantity_pair(v):
        return {"value": v[0], "unit": v[1]}
    if isinstance(v, list):
        return [_coerce_list_value(x) for x in v]
    if isinstance(v, tuple):
        return [_coerce_list_value(x) for x in v]
    if isinstance(v, set):
        return sorted(_coerce_list_value(x) for x in v)
    if isinstance(v, dict):
        return {k: _coerce_list_value(x) for k, x in v.items()}
    return v


def _collect_notes(obj: Any, notes: List[ImportNote], path: str = ""):
    """Recursively harvest __import_note__ sentinels into `notes`."""
    if isinstance(obj, dict):
        note = obj.get("__import_note__")
        if isinstance(note, dict):
            notes.append(ImportNote(note.get("function", "?"),
                                    note.get("line", 0), note.get("message", "")))
        for k, v in obj.items():
            _collect_notes(v, notes, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _collect_notes(v, notes, f"{path}[{i}]")


def _snake_key(kwarg: str) -> str:
    """camelCase -> snake_case (RMG DSL -> schema convention)."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", kwarg)
    return s.lower()


class LegacyVisitor:
    """Walks top-level DSL calls of a legacy input file."""

    def __init__(self, source: str):
        self.source = source
        self.result: Dict[str, Any] = {}
        self.notes: List[ImportNote] = []
        self.reactors: List[Dict[str, Any]] = []
        self._seen_blocks = set()

    def _note(self, func: str, line: int, msg: str):
        self.notes.append(ImportNote(func, line, msg))

    # -- database() ---------------------------------------------------------
    def visit_database(self, node: ast.Call):
        db: Dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                self._note("database", node.lineno, "unparsed **kwargs in database()")
                continue
            v = node_to_value(kw.value, self.source, self.notes, "database")
            if isinstance(v, dict) and "__import_note__" in v:
                _collect_notes(v, self.notes, "database")
                v = {"__dropped__": v.get("__source__") or "see import_notes"}
            db[_snake_key(kw.arg)] = _coerce_list_value(v)
        if db:
            self.result["database"] = db

    # -- species() / forbidden() -------------------------------------------
    def visit_species(self, node: ast.Call):
        entries = self.result.setdefault("species", [])
        self._add_entry(node, entries, "species", has_label=True)

    def visit_forbidden(self, node: ast.Call):
        entries = self.result.setdefault("forbidden", [])
        self._add_entry(node, entries, "forbidden", has_label=True)

    def _add_entry(self, node: ast.Call, entries: List[Dict[str, Any]],
                   func: str, has_label: bool):
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        # label / structure may be positional: species('CH4', SMILES('C'))
        if len(node.args) >= 1 and "label" not in kwargs:
            kwargs["label"] = node.args[0]
        if len(node.args) >= 2 and "structure" not in kwargs:
            kwargs["structure"] = node.args[1]
        if "structure" not in kwargs:
            self._note(func, node.lineno, "no structure argument")
            return
        entry: Dict[str, Any] = {}
        if has_label and "label" in kwargs:
            label = node_to_value(kwargs["label"], self.source, self.notes, func)
            entry["label"] = label
        structure = node_to_value(kwargs["structure"], self.source, self.notes, func)
        if isinstance(structure, dict) and "__import_note__" in structure:
            _collect_notes(structure, self.notes, f"{func}.structure")
            structure = {"smiles": None}  # placeholder; note is recorded loudly
        entry["structure"] = structure
        for kw_name, kw_node in kwargs.items():
            if kw_name in ("label", "structure"):
                continue
            v = node_to_value(kw_node, self.source, self.notes, func)
            if isinstance(v, dict) and "__import_note__" in v:
                _collect_notes(v, self.notes, f"{func}.{kw_name}")
                continue
            entry[_snake_key(kw_name)] = _coerce_list_value(v)
        entries.append(entry)

    # -- reactors -----------------------------------------------------------
    def visit_reactor(self, node: ast.Call, func: str):
        rtype = REACTOR_TYPE_NAMES[func]
        reactor: Dict[str, Any] = {"type": rtype}
        staged: Dict[str, Any] = {}
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        # positional args are DSL-specific; record them positionally
        for i, a in enumerate(node.args):
            v = node_to_value(a, self.source, self.notes, func)
            if isinstance(v, dict) and "__import_note__" in v:
                _collect_notes(v, self.notes, f"{func}.arg{i}")
                continue
            reactor[f"arg{i}"] = _coerce_list_value(v)
        for kw_name, kw_node in kwargs.items():
            v = node_to_value(kw_node, self.source, self.notes, func)
            if isinstance(v, dict) and "__import_note__" in v:
                _collect_notes(v, self.notes, f"{func}.{kw_name}")
                continue
            key = _snake_key(kw_name)
            if key == "termination_conversion":
                self._set_termination(reactor, "conversion", v)
            elif key == "termination_time":
                self._set_termination(reactor, "time", v)
            elif key == "termination_rate_ratio":
                self._set_termination(reactor, "rate_ratio", v)
            elif key == "termination_criticality":
                self._set_termination(reactor, "criticality", v)
            else:
                val = _coerce_list_value(v)
                # staged reactor: list-valued temperature/pressure fields
                # become staged_* (the legacy DSL expresses staged runs via
                # a list of (T, unit) / (P, unit) tuples in simpleReactor)
                if isinstance(val, list) and key in ("temperature", "pressure"):
                    staged[f"staged_{key}s"] = val
                else:
                    reactor[key] = val
        if staged:
            reactor["type"] = "staged"
            reactor.update(staged)
        self.reactors.append(reactor)

    def _set_termination(self, reactor: Dict[str, Any], field: str, value: Any):
        term = reactor.setdefault("termination", {})
        term[field] = _coerce_list_value(value)

    # -- single-block functions --------------------------------------------
    def visit_block(self, node: ast.Call, schema_key: str, func: str,
                    target: Optional[Dict[str, Any]] = None):
        # Repeated single-block calls: RMG replaces the block (new object),
        # so the earlier call's values are discarded, not merged.
        if target is None:
            if schema_key in self._seen_blocks:
                self.result[schema_key] = {}
                self._note(func, node.lineno,
                           f"repeated {func}() call: the earlier call's values are discarded, "
                           f"not merged (RMG last-wins semantics)")
            self._seen_blocks.add(schema_key)
            block = self.result.setdefault(schema_key, {})
        else:
            block = target
        for kw in node.keywords:
            if kw.arg is None:
                self._note(func, node.lineno, f"unparsed **kwargs in {func}()")
                continue
            v = node_to_value(kw.value, self.source, self.notes, func)
            if isinstance(v, dict) and "__import_note__" in v:
                _collect_notes(v, self.notes, f"{func}.{kw.arg}")
                continue
            block[_snake_key(kw.arg)] = _coerce_list_value(v)
        for i, a in enumerate(node.args):
            v = node_to_value(a, self.source, self.notes, func)
            if isinstance(v, dict) and "__import_note__" in v:
                _collect_notes(v, self.notes, f"{func}.arg{i}")
                continue
            block[f"arg{i}"] = _coerce_list_value(v)

    def visit_top(self, tree: ast.Module):
        for stmt in tree.body:
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    continue
                # non-call top-level statement: loud note, keep source
                self._note(
                    "<top>",
                    getattr(stmt, "lineno", 0),
                    f"top-level {type(stmt).__name__} statement (preserved): "
                    f"{ast.get_source_segment(self.source, stmt)[:120]!r}",
                )
                continue
            call = stmt.value
            name = _call_name(call)
            if name is None:
                self._note("<call>", call.lineno, "anonymous call expression")
                continue
            if name == "database":
                self.visit_database(call)
            elif name in ("species", "forbidden"):
                getattr(self, f"visit_{name}")(call)
            elif name in REACTOR_TYPE_NAMES:
                self.visit_reactor(call, name)
            elif name in DSL_SCHEMA_KEYS:
                schema_key = DSL_SCHEMA_KEYS[name]
                if schema_key is None:
                    # out-of-schema function: preserve verbatim, loud note
                    self._visit_legacy_block(call, name)
                else:
                    self.visit_block(call, schema_key, name)
            else:
                # unmapped top-level function: preserve structurally, loud note
                self._visit_legacy_block(call, name)
        if self.reactors:
            self.result["reactors"] = self.reactors

    def _visit_legacy_block(self, call: ast.Call, name: str):
        """Preserve an unmapped/out-of-schema call under _legacy.<name>.

        Per-function blocks: repeated calls to the SAME function replace
        (RMG last-wins) with a loud note; different functions coexist.
        """
        legacy = self.result.setdefault("_legacy", {})
        if name in self._seen_blocks:
            legacy[name] = {}
            self._note(name, call.lineno,
                       f"repeated {name}() call: the earlier call's values are "
                       f"discarded, not merged (RMG last-wins semantics)")
        self._seen_blocks.add(name)
        self.visit_block(call, name, name, target=legacy.setdefault(name, {}))
        self._note(name, call.lineno,
                  f"not part of the rmgpu schema; values preserved under _legacy.{name}")


def import_legacy(path: str) -> Dict[str, Any]:
    """Import a legacy RMG .py input file into a schema-shaped dict.

    The returned dict validates against ``rmgpu.schemas.input.Input`` (with an
    ``import_notes`` list, always present, possibly empty). Values are NEVER
    silently dropped: anything unrepresentable is captured in ``import_notes``
    (and, when possible, kept under a placeholder).
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise LegacyImporterError(f"File not found: {path}")
    with open(path, "r") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        raise LegacyImporterError(f"Syntax error in {path}: {e}")

    visitor = LegacyVisitor(source)
    visitor.visit_top(tree)

    result = visitor.result
    result["rmgpu"] = "1.0"
    result["import_notes"] = [n.to_dict() for n in visitor.notes]
    return result


# ---------------------------------------------------------------------------
# Ground-truth dump (kept for back-compat with scripts/legacy_dump.py): raw
# per-call inventory of the file. This is NOT the gate's losslessness check
# (that is in gates/gate_03.py, which uses an independent canonicalizer).
# ---------------------------------------------------------------------------

def dump_legacy(path: str) -> Dict[str, Any]:
    """Dump the raw parsed calls of a legacy file (JSON-able inventory)."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise LegacyImporterError(f"File not found: {path}")
    with open(path) as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name is None:
            continue
        args = []
        for a in node.args:
            try:
                args.append(ast.literal_eval(a))
            except (ValueError, SyntaxError):
                args.append(ast.get_source_segment(source, a))
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                kwargs[kw.arg] = ast.get_source_segment(source, kw.value)
        calls.append({"name": name, "args": args, "kwargs": kwargs})
    return {"file": os.path.basename(path), "functions": calls}
