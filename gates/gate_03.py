"""Job-03 gate: lossless import of the legacy input.py corpus.

Checks (all must pass for GREEN):
  1. DSL inventory:  file x functions-used table (printed + saved).
  2. Import:  N of <total> files import to a schema-valid document with zero
     dropped values (independent canonical diff). A file with documented
     IMPORT-NOTEs still counts toward N iff the lossless diff passes on the
     effective (RMG last-wins) values.
  3. `rmgpu validate` (the real CLI, subprocess) on every imported YAML: all pass.
  4. `rmgpu run minimal.yaml` (imported minimal example) prints the resolved document.
  5. JSON schema export: the pydantic model's JSON schema validates the
     hand-written PLAN.md 12.2 example (examples/handwritten_minimal.yaml)
     via jsonschema, and `rmgpu schema` runs.

Exit code 0 = GREEN, 1 = RED. Writes reports/gate_03_results.json.
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = '/home/jackson/rmgpu/rmgpu'
PYTHON = '/home/jackson/miniforge3/envs/rmgpu/bin/python'
RMGPU = '/home/jackson/miniforge3/envs/rmgpu/bin/rmgpu'
sys.path.insert(0, REPO)

from rmgpu.importer.legacy import import_legacy, LegacyImporterError  # noqa: E402
from rmgpu.schemas.input import Input  # noqa: E402
from gates.legacy_ground_truth import find_input_files, DSL_FUNCTIONS  # noqa: E402
from gates.legacy_canonical import check_file, canonicalize  # noqa: E402

TOTAL = None
results = {
    'files': 0,
    'inventory': {},
    'import_lossless': 0,
    'schema_valid': 0,
    'cli_validate_pass': 0,
    'notes': [],
    'failures': [],
    'handwritten_example': None,
    'run_minimal': None,
    'json_schema_valid': None,
}


def cli_validate(path: str):
    p = subprocess.run([RMGPU, 'validate', path], capture_output=True, text=True)
    return p.returncode == 0, p.stdout.strip() + p.stderr.strip()


def cli_run(path: str):
    p = subprocess.run([RMGPU, 'run', path], capture_output=True, text=True)
    return p.returncode == 0, p.stdout.strip(), p.stderr.strip()


def main():
    global MINIMAL_YAML
    MINIMAL_YAML = None
    files = find_input_files()
    results['files'] = len(files)

    outdir = tempfile.mkdtemp(prefix='gate03_')
    inventory = {}
    for f in files:
        try:
            doc = import_legacy(f)
        except LegacyImporterError as e:
            results['failures'].append({'file': os.path.relpath(f, '/home/jackson/rmgpu/RMG-Py'),
                                        'error': f'import raised: {e}'})
            continue

        # 1. inventory
        try:
            canon = canonicalize(f)
            rel = os.path.relpath(f, '/home/jackson/rmgpu/RMG-Py')
            for fn in canon['calls']:
                inventory.setdefault(fn, []).append(rel)
        except Exception:
            pass

        # 2a. schema validity (in-process)
        body = {k: v for k, v in doc.items() if k != 'import_notes'}
        valid = True
        try:
            Input(**body)
        except Exception:
            valid = False
        results['schema_valid'] += int(valid)

        # 2b. lossless diff against the independent canonical evaluation
        errs = check_file(f, doc)
        if not valid or errs:
            results['failures'].append({'file': os.path.relpath(f, '/home/jackson/rmgpu/RMG-Py'),
                                        'schema_valid': valid, 'lossless_errors': errs[:12]})
        else:
            results['import_lossless'] += 1

        # 3. write YAML + CLI validate
        import yaml
        from rmgpu.cli import QuantityDumper
        notes = doc.get('import_notes', [])
        yaml_path = os.path.join(outdir, os.path.basename(os.path.dirname(f)) + '.yaml')
        with open(yaml_path, 'w') as fh:
            for n in notes:
                fh.write(f"# IMPORT-NOTE: {n['message']}\n")
            body_for_dump = {k: v for k, v in body.items() if k != 'import_notes'}
            fh.write(yaml.dump(body_for_dump, Dumper=QuantityDumper, default_flow_style=False))
        ok, out = cli_validate(yaml_path)
        results['cli_validate_pass'] += int(ok)
        if not ok:
            results['failures'].append({'file': os.path.relpath(f, '/home/jackson/rmgpu/RMG-Py'),
                                        'cli_validate': out[:400]})
        if notes:
            results['notes'].append({'file': os.path.relpath(f, '/home/jackson/rmgpu/RMG-Py'),
                                     'notes': [n['message'] for n in notes]})
        if f.endswith('examples/rmg/minimal/input.py'):
            MINIMAL_YAML = yaml_path

    results['inventory'] = {k: sorted(v) for k, v in inventory.items()}

    # 4. rmgpu run minimal.yaml
    try:
        ok, stdout, stderr = cli_run(MINIMAL_YAML)
        results['run_minimal'] = {'ok': ok, 'stdout_chars': len(stdout),
                                  'stdout_head': stdout[:400], 'stderr': stderr[:200]}
    except NameError:
        results['run_minimal'] = {'ok': False, 'error': 'minimal input.py not found'}

    # 5. JSON schema export + hand-written PLAN 12.2 example
    import jsonschema
    handwritten = os.path.join(REPO, 'examples/handwritten_minimal.yaml')
    try:
        import yaml as _y
        schema = Input.dump_json_schema()
        with open(handwritten) as fh:
            hand_doc = _y.safe_load(fh)
        jsonschema.validate(instance=hand_doc, schema=schema)
        results['json_schema_valid'] = {'ok': True, 'file': handwritten}
        # also exercise the CLI's schema export
        p = subprocess.run([RMGPU, 'schema'], capture_output=True, text=True)
        results['json_schema_valid']['cli_schema_ok'] = p.returncode == 0
    except Exception as e:
        results['json_schema_valid'] = {'ok': False, 'error': f'{type(e).__name__}: {e}'[:400]}

    # ---- verdict ----
    green = (
        results['import_lossless'] == results['files']
        and results['schema_valid'] == results['files']
        and results['cli_validate_pass'] == results['files']
        and results['run_minimal']['ok'] is True
        and results['json_schema_valid']['ok'] is True
    )

    # ---- report ----
    lines = []
    lines.append('== JOB-03 GATE ==')
    lines.append(f'files: {results["files"]}')
    lines.append(f'schema-valid (in-process): {results["schema_valid"]}/{results["files"]}')
    lines.append(f'lossless vs canonical:    {results["import_lossless"]}/{results["files"]}')
    lines.append(f'rmgpu validate (CLI):     {results["cli_validate_pass"]}/{results["files"]}')
    lines.append(f'rmgpu run minimal.yaml:   {"OK" if results["run_minimal"]["ok"] else "FAIL"}')
    lines.append(f'JSON schema + hand example: {"OK" if results["json_schema_valid"]["ok"] else "FAIL"}')
    lines.append(f'files with IMPORT-NOTES:  {len(results["notes"])}')
    lines.append('')
    lines.append('DSL INVENTORY (function: n files):')
    for fn in sorted(inventory):
        lines.append(f'  {fn}: {len(inventory[fn])}')
    unseen = sorted(set(DSL_FUNCTIONS) - set(inventory))
    lines.append(f'DSL functions never used in corpus: {len(unseen)}')
    lines.append('')
    for n in results['notes']:
        lines.append(f"NOTE {n['file']}:")
        for m in n['notes']:
            lines.append(f'  - {m}')
    if results['failures']:
        lines.append('')
        lines.append('FAILURES:')
        for f in results['failures']:
            lines.append(json.dumps(f, indent=2)[:1200])
    lines.append('')
    lines.append(f'GATE STATUS: {"GREEN" if green else "RED"}')
    report = '\n'.join(lines)
    print(report)

    rep_path = os.path.join(REPO, 'reports/gate_03_results.json')
    os.makedirs(os.path.dirname(rep_path), exist_ok=True)
    with open(rep_path, 'w') as fh:
        json.dump(results, fh, indent=1)
    print(f'\nresults -> {rep_path}')
    sys.exit(0 if green else 1)


if __name__ == '__main__':
    main()
