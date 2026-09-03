#!/usr/bin/env python3
"""RMG-Py reference runs for the job-06 gate (parity baselines).

Runs RMG-Py (env: rmg_env) on the superminimal and c3h4 example inputs, with
pdep OFF (both examples ship no ``pressure_dependence()`` block - matching
rmgpu's HPL-stub state; job-07 turns pdep on in BOTH), and records the core +
edge species/reaction SETS (canonical SMILES keys) to
``gates/baselines/<example>/summary.json``. These are the non-circular
ground truth the gate_06 parity check compares rmgpu's output against.

Run (in the rmg_env conda env, NOT the rmgpu env):
    /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/rmgpy_reference.py
    /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/rmgpy_reference.py --only c3h4
    # or, re-derive a summary.json from an already-completed output dir:
    /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/rmgpy_reference.py \
        --from-output gates/baselines/superminimal --name superminimal

Notes on the RMG-Py model internals (verified against rmg_env):
  - ``species.molecule`` is a LIST of resonance-form molecule graphs
    (Molecule objects). The first entry is the primary (kekulized) form;
    it carries an adjlist that we canonicalize.
  - ``reaction.reactants`` / ``reaction.products`` are lists of Species.
  - After execution the final chemkin file (chem.inp + species_dictionary.txt)
    under the output dir is a faithful record of the final core model
    (19 reactions for superminimal, matching the log), so the --from-output
    path reconstructs the reaction set from it without re-running RMG.

This script is reference-only: it drives the real RMG-Py, it does not import
rmgpu.
"""
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RMG_PY = "/home/jackson/rmgpu/RMG-Py"
BASE = os.path.join(REPO, "gates", "baselines")

EXAMPLES = {
    "superminimal": os.path.join(RMG_PY, "examples", "rmg", "superminimal", "input.py"),
    "c3h4": os.path.join(RMG_PY, "examples", "rmg", "c3h4", "input.py"),
}


# ---------------------------------------------------------------------------
# Canonical keys (must match rmgpu.core.loop.canonical_key exactly)
# ---------------------------------------------------------------------------

def canon_mol(m):
    """Isomorphism-invariant canonical SMILES (explicit H, kekulized,
    sanitized) - the same key gate_05 uses, applied to both sides."""
    from rdkit import Chem
    if not any(a.GetSymbol() == "H" for a in m.GetAtoms()):
        try:
            m = Chem.AddHs(m)
        except Exception:  # noqa: BLE001
            pass
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        Chem.SanitizeMol(m)
    except Exception:  # noqa: BLE001
        pass
    return Chem.MolToSmiles(m)


def species_key(sp):
    """Canonical key of an RMG-Py Species. ``sp.molecule`` is a LIST of
    resonance forms; the first (primary) form is canonicalized."""
    from rmgpy.molecule.molecule import Molecule as RMGMol
    mols = getattr(sp, "molecule", None)
    if isinstance(mols, (list, tuple)):
        mols = mols[0] if mols else None
    if mols is None or not isinstance(mols, RMGMol):
        raise ValueError("species %s has no usable molecule"
                         % getattr(sp, "label", "?"))
    return canon_mol(mols.to_rdkit_mol())


def _adjlist_key(adj):
    """Canonical key of an RMG-format adjacency-list block (explicit H)."""
    from rmgpy.molecule.molecule import Molecule as RMGMol
    m = RMGMol()
    m.from_adjacency_list(adj)
    return canon_mol(m.to_rdkit_mol())


def reaction_key(rxn):
    """Canonical, order-insensitive reaction key: sorted reactant SMILES +
    '>>' + sorted product SMILES."""
    rs = sorted(species_key(s) for s in rxn.reactants)
    ps = sorted(species_key(s) for s in rxn.products)
    return ".".join(rs) + ">>" + ".".join(ps)


def reaction_detail(rxn):
    """Per-reaction record (family/template) for the systematic-divergence
    analysis."""
    family = getattr(rxn, "family", None)
    template = getattr(rxn, "template", None)
    return {
        "key": reaction_key(rxn),
        "family": family.label if family is not None else None,
        "template": list(template) if template else None,
    }


def _n_iterations_from_xls(outdir):
    """Final model-iteration number from RMG's statistics.xls (iteration
    column of the last row), or None if unreadable."""
    path = os.path.join(outdir, "statistics.xls")
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd
        df = pd.read_excel(path)
        cols = [c for c in df.columns if "iteration" in str(c).lower()]
        if not cols:
            return None
        vals = df[cols[0]].dropna().tolist()
        return int(vals[-1]) if vals else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Live-run path
# ---------------------------------------------------------------------------

def run_example(name, input_py):
    from rmgpy.rmg.main import RMG

    outdir = os.path.join(BASE, name)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    rmg = RMG(input_file=input_py, output_directory=outdir)
    rmg.execute()
    dt = time.time() - t0
    model = rmg.reaction_model
    _summarize_model(model, name, outdir, dt)
    return _read_summary(name)


def _summarize_model(model, name, outdir, dt):
    core_species = sorted(species_key(s) for s in model.core.species)
    edge_species = sorted(species_key(s) for s in model.edge.species)
    core_rxn = [reaction_detail(r) for r in model.core.reactions]
    edge_rxn = [reaction_detail(r) for r in model.edge.reactions]
    core_rxn_keys = sorted({r["key"] for r in core_rxn})
    edge_rxn_keys = sorted({r["key"] for r in edge_rxn})
    summary = {
        "example": name,
        "generator": "RMG-Py",
        "elapsed_seconds": round(dt, 1),
        "n_iterations": getattr(model, "iteration_num", None),
        "core_species": core_species,
        "edge_species": edge_species,
        "core_reactions_keys": core_rxn_keys,
        "edge_reactions_keys": edge_rxn_keys,
        "core_reactions": core_rxn,
        "edge_reactions": edge_rxn,
        "n_core_species": len(core_species),
        "n_edge_species": len(edge_species),
        "n_core_reactions": len(core_rxn_keys),
        "n_edge_reactions": len(edge_rxn_keys),
    }
    path = os.path.join(outdir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print("[rmgpy_reference] %s done in %.0fs: core %d spc/%d rxn, edge %d spc/%d rxn -> %s"
          % (name, dt, len(core_species), len(core_rxn_keys),
             len(edge_species), len(edge_rxn_keys), path))


def _read_summary(name):
    with open(os.path.join(BASE, name, "summary.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# From-output path (reconstruct from a completed RMG output directory)
# ---------------------------------------------------------------------------

def _parse_chemkin(inp_path, annotated_path=None):
    """Parse a chemkin .inp: returns (reactions, species_names). Species
    names are the parenthetical chemkin labels (e.g. ``H2(1)``, ``[O]O(4)``)
    which match species_dictionary.txt verbatim. ``reactions`` is a list of
    (reactants, products) name lists. Optional annotated .inp supplies the
    per-reaction family comment."""
    with open(inp_path) as f:
        lines = f.read().splitlines()

    # species names come from the SPECIES block (authoritative order)
    species_names, in_species = [], False
    for ln in lines:
        s = ln.strip()
        if s == "SPECIES":
            in_species = True
            continue
        if in_species:
            if s == "END":
                in_species = False
                continue
            if s:
                species_names.append(s)

    # reaction lines: the stoichiometry is ONE whitespace token
    # "A+B<=>C+D"; the Arrhenius params follow as separate tokens.
    sp_tok = re.compile(r"^\[?[\w()\[\]]+\]?$")
    reactions = []
    for ln in lines:
        s = ln.strip()
        if "<=>" not in s or s.startswith("!"):
            continue
        head = s.split()[0]              # the "A+B<=>C+D" token
        if "<=>" not in head:
            continue
        lhs, rhs = head.split("<=>", 1)
        rs = [t for t in lhs.split("+") if t]
        ps = [t for t in rhs.split("+") if t]
        if rs and ps and all(sp_tok.match(t) for t in rs + ps):
            reactions.append((rs, ps))

    # family map from the annotated file (a "family: X" comment line
    # precedes each reaction line, within a few lines)
    family = {}
    if annotated_path and os.path.exists(annotated_path):
        with open(annotated_path) as f:
            alines = f.read().splitlines()
        i = 0
        while i < len(alines):
            s = alines[i].strip()
            if "<=>" in s and not s.startswith("!"):
                head = s.split()[0]
                if "<=>" in head:
                    fam = None
                    for j in range(i - 1, max(i - 8, -1), -1):
                        m = re.search(r"family:\s*(\S+)", alines[j])
                        if m:
                            fam = m.group(1)
                            break
                    key = "<=>".join(head.split("<=>")[0].split("+")) + "=>" + \
                          "<=>".join(head.split("<=>")[1].split("+"))
                    family[key] = fam
            i += 1
    return reactions, species_names, family


def _load_dictionary(path):
    """species_dictionary.txt -> {name: adjlist-text}."""
    out = {}
    with open(path) as f:
        txt = f.read()
    for b in re.split(r"\n\s*\n", txt.strip()):
        lines = b.strip().splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        adj = "\n".join(l for l in lines[1:]
                        if not l.strip().startswith("multiplicity"))
        if name and adj.strip():
            out[name] = adj
    return out


def from_output(name, outdir):
    from rdkit import Chem

    ck = os.path.join(outdir, "chemkin")
    inp = os.path.join(ck, "chem.inp")
    ann = os.path.join(ck, "chem_annotated.inp")
    dpath = os.path.join(ck, "species_dictionary.txt")
    if not (os.path.exists(inp) and os.path.exists(dpath)):
        raise SystemExit("from-output: missing chem.inp or species_dictionary.txt in %s" % ck)

    reactions, species_names, fam_map = _parse_chemkin(inp, ann)
    dict_adj = _load_dictionary(dpath)

    name_to_key = {}
    for nm in species_names:
        if nm in dict_adj:
            try:
                name_to_key[nm] = _adjlist_key(dict_adj[nm])
            except Exception as e:  # noqa: BLE001
                print("[rmgpy_reference] WARN: cannot canonicalize %s: %s" % (nm, e),
                      file=sys.stderr)

    core_species = sorted(name_to_key[nm] for nm in species_names
                          if nm in name_to_key)

    rxns = []
    for rs, ps in reactions:
        rs_keys = [name_to_key[r] for r in rs if r in name_to_key]
        ps_keys = [name_to_key[p] for p in ps if p in name_to_key]
        if len(rs_keys) != len(rs) or len(ps_keys) != len(ps):
            continue  # a species missing from the dictionary
        key = ".".join(sorted(rs_keys)) + ">>" + ".".join(sorted(ps_keys))
        fam = fam_map.get("<=>".join(rs) + "=>" + "<=>".join(ps))
        rxns.append({"key": key, "reactants": sorted(rs_keys),
                     "products": sorted(ps_keys), "family": fam})
    keys = sorted({r["key"] for r in rxns})

    summary = {
        "example": name,
        "generator": "RMG-Py (from completed output dir, chem.inp + species_dictionary.txt)",
        "n_iterations": _n_iterations_from_xls(outdir),
        "core_species": core_species,
        "edge_species": [],
        "core_reactions_keys": keys,
        "edge_reactions_keys": [],
        "core_reactions": rxns,
        "edge_reactions": [],
        "n_core_species": len(core_species),
        "n_edge_species": 0,
        "n_core_reactions": len(keys),
        "n_edge_reactions": 0,
    }
    path = os.path.join(outdir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print("[rmgpy_reference] %s from-output: core %d spc / %d rxn (iterations=%s) -> %s"
          % (name, len(core_species), len(keys), summary["n_iterations"], path))
    return summary


# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        only = args[args.index("--only") + 1]
    if "--from-output" in args:
        i = args.index("--from-output")
        outdir = os.path.abspath(args[i + 1])
        nm = None
        if "--name" in args:
            nm = args[args.index("--name") + 1]
        else:
            nm = os.path.basename(outdir.rstrip("/"))
        from_output(nm, outdir)
        return
    for name, path in EXAMPLES.items():
        if only and name != only:
            continue
        if not os.path.exists(path):
            print("[rmgpy_reference] MISSING input: %s" % path, file=sys.stderr)
            continue
        run_example(name, path)
    print("[rmgpy_reference] all done")


if __name__ == "__main__":
    main()
