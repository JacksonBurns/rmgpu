"""Independent canonical evaluation of legacy input.py files (gate ground truth).

This module is the gate's REFERENCE: it evaluates a legacy file's DSL calls
into canonical Python values WITHOUT using rmgpu.importer. The gate then
compares these values against the importer's output document to prove
losslessness (structure equality on values; ordering preserved where the DSL
is order-sensitive: species/forbidden/reactor lists and list items).

The value evaluator is intentionally a separate, minimal implementation
(literals + numeric arithmetic + structure-helper calls); it shares no code
with rmgpu.importer.legacy.
"""
import ast
import os
import re
from typing import Any, Dict, List, Optional

ROOT = '/home/jackson/rmgpu/RMG-Py'

DSL_BLOCKS = {
    'database', 'simulator', 'model', 'pressureDependence', 'mlEstimator',
    'solvation', 'uncertainty', 'options', 'generatedSpeciesConstraints',
    'catalystProperties', 'quantumMechanics',
}

LIST_BLOCKS = {'species', 'forbidden'}

REACTOR_TYPES = {
    'simpleReactor': 'simple',
    'constantVIdealGasReactor': 'const_V',
    'constantTPIdealGasReactor': 'const_TP',
    'liquidReactor': 'liquid',
    'mbsampledReactor': 'mb_sampled',
    'surfaceReactor': 'surface',
    'constantTVLiquidReactor': 'liquid',
    'liquidSurfaceReactor': 'liquid_surface',
}

STRUCTURE = {
    'SMILES': 'smiles',
    'InChI': 'inchi',
    'adjacencyList': 'adjlist',
    'adjacencyListGroup': 'group_adjlist',
    'fragment_adj': 'fragment_adjlist',
    'fragment_SMILES': 'smiles',
    'SMARTS': 'smarts',
}


def find_input_files(root: str = ROOT) -> List[str]:
    files = []
    for sub in ('examples/rmg', 'test/regression'):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            if 'input.py' in filenames:
                files.append(os.path.join(dirpath, 'input.py'))
    return sorted(files)


def snake(k: str) -> str:
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', k).lower()


def eval_value(node: ast.AST, source: str) -> Any:
    """Canonical value of an argument node (independent evaluator)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if k is None:
                continue
            out[str(eval_value(k, source))] = eval_value(v, source)
        return out
    if isinstance(node, (ast.List, ast.Tuple)):
        return [eval_value(e, source) for e in node.elts]
    if isinstance(node, ast.Set):
        return eval_value(ast.copy_location(ast.List(elts=node.elts, ctx=ast.Load()), node), source)
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else None)
        if name in STRUCTURE and node.args:
            inner = eval_value(node.args[0], source)
            if isinstance(inner, str):
                return {'__structure__': name, 'field': STRUCTURE[name], 'value': inner}
        # generic call: structural capture
        return {'__call__': name,
                'args': [eval_value(a, source) for a in node.args],
                'kwargs': {kw.arg: eval_value(kw.value, source)
                           for kw in node.keywords if kw.arg is not None}}
    if isinstance(node, ast.BinOp):
        l, r = eval_value(node.left, source), eval_value(node.right, source)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Div):
            return l / r
    if isinstance(node, ast.UnaryOp):
        v = eval_value(node.operand, source)
        return v if isinstance(node.op, ast.UAdd) else -v
    raise ValueError(f"cannot evaluate node {type(node).__name__}: "
                     f"{ast.get_source_segment(source, node)[:80]!r}")


def canonicalize(path: str) -> Dict[str, Any]:
    """Evaluate the whole file into a canonical call inventory.

    Returns {'calls': {func: [call_dict, ...]}, 'inventory': [funcs in file order]}.
    Each call_dict = {'args': [...], 'kwargs': {name: value}} with values
    already evaluated (structure calls become {'__structure__': ...}).
    """
    with open(path) as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    calls: Dict[str, List[Dict[str, Any]]] = {}
    order = []
    for stmt in tree.body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)):
            raise ValueError(f"unexpected top-level statement at line {stmt.lineno}")
        name = stmt.value.func.id
        call = {
            'line': stmt.value.lineno,
            'args': [eval_value(a, source) for a in stmt.value.args],
            'kwargs': {kw.arg: eval_value(kw.value, source)
                       for kw in stmt.value.keywords if kw.arg is not None},
        }
        calls.setdefault(name, []).append(call)
        order.append(name)
    return {'file': os.path.relpath(path, ROOT), 'calls': calls, 'order': order}


# ---------------------------------------------------------------------------
# Value normalization: canonical value -> form expected in the imported doc
# ---------------------------------------------------------------------------

def norm_value(v: Any) -> Any:
    """Normalize a canonical value to the form the importer must emit."""
    if isinstance(v, dict):
        if '__structure__' in v:
            return {v['field']: v['value']}
        if '__call__' in v:
            return v
        # quantity tuple? handled after list conversion
        return {k: norm_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        lst = [norm_value(x) for x in v]
        # (number, unit-string) pair -> quantity dict
        if len(lst) == 2 and isinstance(lst[0], (int, float)) \
                and not isinstance(lst[0], bool) and isinstance(lst[1], str):
            return {'value': lst[0], 'unit': lst[1]}
        return lst
    return v


def compare_values(expected: Any, actual: Any, path: str, errors: List[str]):
    """Structural equality on values (order-sensitive for lists)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            missing = set(expected) - set(actual)
            extra = set(actual) - set(expected)
            if missing:
                errors.append(f"{path}: missing keys {sorted(missing)}")
            if extra:
                errors.append(f"{path}: extra keys {sorted(extra)}")
        for k in set(expected) & set(actual):
            compare_values(expected[k], actual[k], f"{path}.{k}", errors)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            errors.append(f"{path}: list length {len(expected)} != {len(actual)}")
        for i, (e, a) in enumerate(zip(expected, actual)):
            compare_values(e, a, f"{path}[{i}]", errors)
    else:
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                if abs(float(expected) - float(actual)) > 1e-12:
                    errors.append(f"{path}: {expected!r} != {actual!r}")
                return
            except (TypeError, ValueError):
                pass
        if expected != actual:
            errors.append(f"{path}: {expected!r} != {actual!r}")


def check_file(path: str, doc: Dict[str, Any]) -> List[str]:
    """Compare the canonical file against the importer's document.

    `doc` is the importer's output (with import_notes removed). Returns a list
    of error strings (empty == lossless).
    """
    errors: List[str] = []
    canon = canonicalize(path)

    for func, calls in canon['calls'].items():
        if func in DSL_BLOCKS:
            # RMG is last-wins for single-block functions; the effective call
            # is the last one, and the override must be documented.
            if len(calls) > 1:
                note_text = ' '.join(str(n) for n in doc.get('import_notes', []))
                if func not in note_text:
                    errors.append(f"{func}: {len(calls)} calls but no IMPORT-NOTE documents the override")
            effective = [calls[-1]]
        else:
            effective = calls

        for i, call in enumerate(effective):
            exp = _expected_block(doc, func, call, i)
            if exp is None:
                continue  # nothing to compare (documented separately)
            if func in LIST_BLOCKS:
                actual = doc.get(func, [])
                if i >= len(actual):
                    errors.append(f"{func}[{i}]: missing in document (has {len(actual)})")
                    continue
                for k, v in exp.items():
                    compare_values(v, actual[i].get(k), f"{func}[{i}].{k}", errors)
                for k in actual[i]:
                    if k not in exp:
                        errors.append(f"{func}[{i}]: unexpected key {k!r}")
            elif func in REACTOR_TYPES:
                actual = doc.get('reactors', [])
                if i >= len(actual):
                    errors.append(f"reactors[{i}] (from {func}): missing in document")
                    continue
                for k, v in exp.items():
                    compare_values(v, actual[i].get(k), f"reactors[{i}].{k}", errors)
                for k in actual[i]:
                    if k not in exp:
                        errors.append(f"reactors[{i}]: unexpected key {k!r}")
            else:
                schema_key = DSL_BLOCKS_KEY.get(func)
                actual = doc.get(schema_key, {})
                for k, v in exp.items():
                    compare_values(v, actual.get(k), f"{schema_key}.{k}", errors)
                for k in actual:
                    if k not in exp and not k.startswith('arg'):
                        errors.append(f"{schema_key}: unexpected key {k!r}")
    return errors


DSL_BLOCKS_KEY = {
    'database': 'database',
    'simulator': 'simulator',
    'model': 'model',
    'pressureDependence': 'pressure_dependence',
    'mlEstimator': 'ml_estimator',
    'solvation': 'solvation',
    'uncertainty': 'uncertainty',
    'options': 'options',
    'generatedSpeciesConstraints': 'generated_species_constraints',
    'catalystProperties': 'catalyst_properties',
    'quantumMechanics': 'quantum_mechanics',
}


def _expected_block(doc: Dict[str, Any], func: str, call: Dict[str, Any],
                    index: int) -> Optional[Dict[str, Any]]:
    """Build the expected (snake_case key -> canonical value) mapping for a call."""
    exp: Dict[str, Any] = {}
    if func in REACTOR_TYPES:
        rtype = REACTOR_TYPES[func]
        kwargs = call['kwargs']
        for k, v in kwargs.items():
            sk = snake(k)
            if sk in ('termination_conversion', 'termination_time',
                      'termination_rate_ratio', 'termination_criticality'):
                field = sk[len('termination_'):]
                term = exp.setdefault('termination', {})
                term[field] = norm_value(v)
            elif sk in ('temperature', 'pressure'):
                nv = norm_value(v)
                # staged: a list of (value, unit) quantity pairs
                if isinstance(nv, list) and len(nv) >= 2 and all(
                        isinstance(x, dict) and set(x) == {'value', 'unit'} for x in nv):
                    exp[f'staged_{sk}s'] = nv
                    exp['type'] = 'staged'
                else:
                    exp[sk] = nv
            else:
                exp[sk] = norm_value(v)
        if 'type' not in exp:
            exp['type'] = rtype
        for j, a in enumerate(call['args']):
            exp[f'arg{j}'] = norm_value(a)
        return exp
    if func in ('species', 'forbidden'):
        kwargs = dict(call['kwargs'])
        if len(call['args']) >= 1 and 'label' not in kwargs:
            kwargs['label'] = call['args'][0]
        if len(call['args']) >= 2 and 'structure' not in kwargs:
            kwargs['structure'] = call['args'][1]
        for k, v in kwargs.items():
            exp[snake(k)] = norm_value(v)
        return exp
    # block functions
    for k, v in call['kwargs'].items():
        exp[snake(k)] = norm_value(v)
    for j, a in enumerate(call['args']):
        exp[f'arg{j}'] = norm_value(a)
    return exp
