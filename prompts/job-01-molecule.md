# job-01: Units + molecule layer

Read ORIENTATION.md and PLAN.md sections 5 (delegations), 8 (layout) first.
Prereq: job-00 done (env `rmgpu`, package skeleton).

## Goal

The molecular layer: everything that represents a chemical structure. Internal
representation is RDKit; the RMG adjacency list is a serialization format. This is
the substrate every later job uses, so correctness and clean APIs matter more than
speed here.

## Reference code (read before porting)

  RMG-Py/rmgpy/molecule/
    translator.py    (21k)  RDKit <-> Molecule conversion - the core of our wrapper
    adjlist.py       (51k)  adjacency-list parse/serialize
    molecule.py      (133k) Molecule class - port the SEMANTICS (charges, radicals,
                            atom types, symmetry numbers, resonance handling), NOT the
                            hand-rolled graph code (that is RDKit's job now)
    atomtype.py      (97k)  atom-type DB loading + assignment
    atomtypedatabase.py (6k)
    element.py       (22k)  element table incl. symbol_by_number
    symmetry.py      (25k)  symmetry numbers / optical isomers (used by statmech later)
    resonance.py     (61k)  resonance structure generation rules
    filtration.py    (26k)  forbidden-structure / filter matching (SMARTS-based)
    inchi.py, inchi handling -> delegate to RDKit
    converter.py, pathfinder.py, group.py -> READ to understand what is needed;
                            port only what later jobs require (groups used in job 05)

## Deliverables

1. `rmgpu/units.py`
   - `Quantity` thin over pint: value + unit, SI-internal (J, K, Pa, mol, kg).
   - constructors from (value, unit) tuples and from "1350 K" strings (used by the
     importer); `.to_si()`, arithmetic with unit checking.
   - this replaces the cimported RMG Quantity everywhere.

2. `rmgpu/molecule/molecule.py`
   - `Molecule` = wrapper holding: `rdkit: RWMol` (canonical, kekulized), charge,
     radical count, atom-type assignments, labels (labeled atoms for recipes),
     resonance structure list, formula, symmetry number.
   - constructors: from SMILES, from InChI, from adjacency list, from an RWMol.
   - `.to_smiles()`, `.to_adjlist()`, `.to_inchi()`, `__eq__`/`__hash__` via RDKit
     canonical form (isomorphism = RDKit, NOT a re-implemented VF2),
     `.is_isomorph(other)`, `.is_substructure(other)`, `.substructure_match_count(other)`
     (degeneracy counting, via RDKit GetSubstructMatch / GetSubstructMatches),
     `.get_formula()`, `.get_symmetry_number()`, charge/radical accessors.
   - atom-labeled atoms (needed by recipes in job 05): keep RMG's labeled-atom
     semantics (labels like '1','2','*') as a side map.

3. `rmgpu/molecule/adjlist.py`
   - parse/serialize of RMG's adjacency-list format, byte-compatible with RMG-Py
     output (this format is the species_dictionary interchange and appears in
     libraries and examples). Reference: molecule/adjlist.py.

4. `rmgpu/molecule/atomtype.py`
   - load the atom-type DB (RMG-database/input/atomtypes/... find the actual path -
     grep RMG-Py for atomtype database loading) and implement `.assign_atom_types()`
     on Molecule. Atom types are RMG-specific patterns over the graph; keep RMG's
     definition files as data, rewrite the matcher against RDKit.

5. `rmgpu/molecule/resonance.py`
   - port resonance structure generation (RMG's rules, which differ from RDKit's).
     This stays in rmgpu (PLAN.md retain #3).

6. `rmgpu/molecule/symmetry.py`
   - symmetry numbers + optical isomer counting (port semantics; used by statmech in
     job 07 for conformer factors).

## Gate (job 01) -> gates/gate_01.py, report reports/job-01.md

For every molecule in these sets (generate the list from: species in
examples/rmg/superminimal, c3h4, minimal input.py files + all entries of a small
thermo library):
  1. adjlist round-trip: parse(Molecule.to_adjlist()) == Molecule (structural
     equality via RDKit isomorphism), and to_adjlist() string-equal to RMG-Py's
     toAdjlist() for the same molecule (run a small RMG-Py script to generate the
     reference strings; save to gates/baselines/adjlist/).
  2. canonical SMILES: rmgpu Molecule.to_smiles() == RDKit canonical (sanity).
  3. isomorphism/substructure on a 20-pair set (including the classic "same molecule
     different atom ordering" and substructure-with-degeneracy cases): results must
     match RDKit direct calls (this is now trivially true - the point is to pin the
     API and catch wrapper bugs).
  4. atom types: for each test molecule, assign_atom_types() must equal RMG-Py's
     assignment (generate reference via RMG-Py script; save baselines).
Record counts (molecules tested, pass/fail per check) in the report.

## When done

Update STATUS.md (job 01 row + session log entry), commit "job-01: units + molecule
layer", STOP.

## Pitfalls

- Do NOT port molecule/graph.pyx, vf2.pyx, kekulize.pyx - those are replaced by
  RDKit. If you find yourself porting graph operations, stop and use RDKit.
- The adjacency list format is fussy (isotope notation, charge syntax, bond symbols,
  radical electrons, stereo). Port the parser strictly; keep RMG-Py's test
  fixtures (RMG-Py/tests/rmgpy/molecule/test_adjlist.py) as test cases - copy what
  you need into rmgpu/tests/ with attribution.
- atomtype.py is 97KB because of the DB loader; the DB itself lives in
  RMG-database - locate it and treat it as data (may become an rmgdb table in job 02;
  until then read the YAML directly).
- Keep Molecule immutable-by-default (copy-on-write) - the recipe engine (job 05)
  mutates bond labels on copies.
