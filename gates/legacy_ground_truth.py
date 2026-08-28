"""Ground-truth AST dump of legacy input.py files.

For every top-level DSL call, records the function name and the FULL source
text of each positional argument and keyword argument (via ast.get_source_segment).
This is lossless w.r.t. literals and call expressions alike: it captures exactly
what the file says, and is the reference the gate diffs the imported document
against.

Also collects, per file, the set of all literal leaves (str/int/float/bool/None,
list/dict items) so the gate can verify "zero dropped values": every literal leaf
of every call must appear in the imported document.
"""
import ast
import json
import os
import sys

DSL_FUNCTIONS = [
    'database', 'thermoLibraries', 'reactionLibraries', 'seedMechanisms',
    'kineticsFamilies', 'kineticsDepositories', 'kineticsEstimator',
    'transportLibraries',
    'species', 'forbidden', 'SMILES', 'InChI', 'adjacencyList',
    'adjacencyListGroup', 'SMARTS', 'fragment_SMILES', 'fragment_adj',
    'react', 'coreSpeciesFile',
    'simpleReactor', 'constantVIdealGasReactor', 'constantTPIdealGasReactor',
    'liquidReactor', 'mbsampledReactor', 'surfaceReactor',
    'constantTVLiquidReactor', 'liquidSurfaceReactor',
    'simulator', 'solvation', 'model', 'quantumMechanics', 'mlEstimator',
    'pressureDependence', 'options', 'generatedSpeciesConstraints',
    'uncertainty', 'restartFromSeed', 'catalystProperties',
    'liquidVolumetricMassTransferCoefficientPowerLaw',
    'SolventData',
]


def find_input_files(root='/home/jackson/rmgpu/RMG-Py'):
    files = []
    for sub in ('examples/rmg', 'test/regression'):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            if 'input.py' in filenames:
                files.append(os.path.join(dirpath, 'input.py'))
    return sorted(files)


def collect_literals(node, out):
    """Collect every literal leaf value under an AST node."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant):
            out.append(child.value)
        elif isinstance(child, (ast.List, ast.Tuple)):
            pass  # constants inside are walked too
        elif isinstance(child, (ast.Call, ast.Name, ast.Attribute)):
            pass  # not literals; calls/identifiers handled structurally


def dump_file(path):
    with open(path) as f:
        source = f.read()
    tree = ast.parse(source, filename=path)

    calls = []
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name):
            continue
        name = call.func.id
        args = []
        for a in call.args:
            seg = ast.get_source_segment(source, a)
            args.append(seg if seg is not None else repr(ast.dump(a)))
        kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                continue
            seg = ast.get_source_segment(source, kw.value)
            kwargs[kw.arg] = seg if seg is not None else repr(ast.dump(kw.value))
        calls.append({'name': name, 'args': args, 'kwargs': kwargs, 'line': call.lineno})

    # Literal leaves of the WHOLE file (everything the importer must not drop)
    leaves = []
    collect_literals(tree, leaves)

    used = sorted({c['name'] for c in calls if c['name'] in DSL_FUNCTIONS})
    unknown = sorted({c['name'] for c in calls if c['name'] not in DSL_FUNCTIONS})

    return {
        'file': os.path.relpath(path, '/home/jackson/rmgpu/RMG-Py'),
        'calls': calls,
        'literals': leaves,
        'dsl_functions': used,
        'unknown_calls': unknown,
    }


def main():
    files = find_input_files()
    out = {'files': [], 'inventory': {}}
    for f in files:
        d = dump_file(f)
        out['files'].append(d)
        for fn in d['dsl_functions']:
            out['inventory'].setdefault(fn, []).append(d['file'])

    dest = sys.argv[1] if len(sys.argv) > 1 else '/tmp/legacy_ground_truth.json'
    with open(dest, 'w') as fh:
        json.dump(out, fh, indent=1)

    print(f"{len(files)} files -> {dest}")
    print("\nDSL INVENTORY (function -> n files):")
    for fn in sorted(out['inventory']):
        print(f"  {fn}: {len(out['inventory'][fn])}")
    print("\nUnknown (non-DSL) calls seen:")
    for d in out['files']:
        if d['unknown_calls']:
            print(f"  {d['file']}: {d['unknown_calls']}")
    # Files whose calls include functions NOT in DSL_FUNCTIONS at all
    all_names = set()
    for d in out['files']:
        all_names |= {c['name'] for c in d['calls']}
    missing = sorted(all_names - set(DSL_FUNCTIONS))
    print(f"\nCall names never seen as DSL: {missing}")


if __name__ == '__main__':
    main()
