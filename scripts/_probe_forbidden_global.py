#!/usr/bin/env python3
"""Validate the superminimal O-chain fix (RMG-Py-faithful).

Root cause: rmgpu over-promotes long O-chains (O4..O16, HO4.., H2O4..) into the
core because it never applies the GLOBAL forbidden-structures file. RMG-Py does
(model.py:1221 add_species_to_core + the reactor prune). The global
forbiddenStructures.py has group entries O3 (forbids any 3-O chain) and O4..
(forbids 4-O chains) whose SUBGRAPH match catches every O-chain with >=3 O.

This probe mirrors RMG-Py's ForbiddenStructures.is_molecule_forbidden EXACTLY
(base.py): it parses the global file's THREE entry types -
  molecule=/species=  -> concrete Molecule -> ISOMORPHISM check
  group=              -> Group             -> SUBGRAPH isomorphism check
- and reports, for a set of known species, which are forbidden.

Expected: the O-chains are forbidden; the real superminimal core species
(H2, O2, H, OH, HO2, H2O, H2O2, O, O2-singlet, and the inerts) are NOT.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FORBIDDEN_PY = "/home/jackson/rmgpu/RMG-database/input/forbiddenStructures.py"
CORE_YAML = os.path.join(REPO, "examples", "run_output_sm", "mechanism", "core.yaml")


def parse_global_forbidden(path):
    """Exec the global forbiddenStructures.py in a controlled namespace and
    return a list of (kind, item) where kind is 'molecule' (item=a Molecule)
    or 'group' (item=a Group), mirroring RMG-Py's load_entry which accepts
    group=None, molecule=None, species=None (species==molecule for our
    isomorphism check)."""
    from rmgpu.molecule.molecule import Molecule
    from rmgpu.molecule.group import Group

    entries = []

    def entry(*, label="", group=None, molecule=None, species=None,
              shortDesc="", longDesc="", **kw):
        entries.append({"label": label, "group": group, "molecule": molecule,
                        "species": species})

    env = {"entry": entry, "True": True, "False": False, "None": None}
    with open(path) as f:
        src = f.read()
    g = dict(env)
    exec(compile(src, path, "exec"), g)

    out = []
    n_mol = n_grp = n_skip = 0
    for e in entries:
        label = e["label"]
        text_mol = (e["molecule"] or e["species"])
        text_grp = e["group"]
        if text_mol:
            try:
                out.append(("molecule", Molecule.from_adjacency_list(
                    text_mol.strip(), saturate_h=True)))
                n_mol += 1
            except Exception as ex:  # noqa: BLE001
                print("  WARN molecule %r: %s" % (label, ex))
                n_skip += 1
        elif text_grp:
            text = text_grp.strip()
            if text.startswith(("OR{", "AND{", "NOT OR{", "NOT AND{")):
                n_skip += 1
                continue  # logic node (none expected in this file)
            try:
                out.append(("group", Group().parse(text)))
                n_grp += 1
            except Exception as ex:  # noqa: BLE001
                print("  WARN group %r: %s" % (label, ex))
                n_skip += 1
        else:
            n_skip += 1
    print("global forbidden: %d total, %d molecule(isomorphism), "
          "%d group(subgraph), %d skipped"
          % (len(entries), n_mol, n_grp, n_skip))
    return out


def is_molecule_forbidden(molecule, entries):
    """RMG-Py ForbiddenStructures.is_molecule_forbidden (base.py)."""
    from rmgpu.core import enumeration as enum
    from rmgpu.core.enumeration import _clean_explicit
    mol = _clean_explicit(molecule)
    for kind, item in entries:
        if kind == "molecule":
            # isomorphism check
            try:
                if _is_isomorphic(mol, item):
                    return True
            except Exception:  # noqa: BLE001
                pass
        else:  # group -> subgraph
            if enum.is_molecule_forbidden(mol, [item]):
                return True
    return False


def _is_isomorphic(a, b):
    """Exact graph isomorphism (RMG Molecule.is_isomorphic), explicit-H,
    stereochemistry ignored. Canonical non-isomeric SMILES with explicit Hs
    is a faithful equality test for these small molecules."""
    from rdkit import Chem
    def canon(m):
        m = Chem.Mol(m._rdkit)
        try:
            Chem.SanitizeMol(m)
        except Exception:  # noqa: BLE001
            pass
        if not any(x.GetSymbol() == "H" for x in m.GetAtoms()):
            m = Chem.AddHs(m)
        return Chem.MolToSmiles(m, isomericSmiles=False)
    return canon(a) == canon(b)


def main():
    import yaml
    from rmgpu.molecule.molecule import Molecule

    entries = parse_global_forbidden(FORBIDDEN_PY)

    # --- (A) validation ---
    ref_ok = {  # real superminimal core species: must be NOT forbidden
        "[H][H]": "H2", "O=O": "O2(singlet)", "[Ar]": "Ar", "[H]": "H",
        "[H]OO[H]": "H2O2", "[H]O[H]": "H2O", "[H]O[O]": "HO2",
        "[H][O]": "OH", "[He]": "He", "[Ne]": "Ne", "[O]": "O",
        "[O][O]": "O2(triplet)", "N#N": "N2",
    }
    ref_bad = {  # O-chains rmgpu wrongly keeps: MUST be forbidden
        "[O]OO[O]": "O4", "[H]OOOOO[O]": "HO6", "[H]OOOO[H]": "H2O4",
        "[O]OOOOOOOOOOOOOO[O]": "O16", "[H]OOOOOOOOOOO[O]": "HO12",
        "O=O-O": "O3 (the O3 group's own case)",
    }
    print("\n=== (A) validation ===")
    n_bad = 0
    for sm, f in sorted(ref_ok.items()):
        forb = is_molecule_forbidden(Molecule(smiles=sm), entries)
        if forb:
            n_bad += 1
            print("  WRONG-FORBIDDEN  %-12s %s" % (sm, f))
        else:
            print("  ok                 %-12s %s" % (sm, f))
    for sm, f in sorted(ref_bad.items()):
        try:
            forb = is_molecule_forbidden(Molecule(smiles=sm), entries)
        except Exception as ex:  # noqa: BLE001
            print("  ERROR %-14s %s: %s" % (sm, f, ex))
            n_bad += 1
            continue
        if not forb:
            n_bad += 1
            print("  WRONG-OK (want forbidden) %-12s %s" % (sm, f))
        else:
            print("  forbidden (expected)    %-12s %s" % (sm, f))
    print("  -> %d validation failures (want 0)" % n_bad)

    # --- (B) the current rmgpu core.yaml species ---
    art = yaml.safe_load(open(CORE_YAML))
    spcs = art["core"]["species"]
    print("\n=== (B) current rmgpu core.yaml species ===")
    n_forbidden = 0
    for s in spcs:
        forb = is_molecule_forbidden(Molecule(smiles=s["smiles"]), entries)
        if forb:
            n_forbidden += 1
        print("  %-10s %-24s %-8s" % ("FORBIDDEN" if forb else "ok       ",
                                      s["smiles"], s.get("formula", "")))
    print("  -> %d of %d core species forbidden (want: only the O-chains)"
          % (n_forbidden, len(spcs)))


if __name__ == "__main__":
    main()
