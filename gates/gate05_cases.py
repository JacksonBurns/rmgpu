"""
job-05/step-05 gate case set.

The job-05 gate runs over a fixed set of (family, reactants) cases:
every family in the 'default' set that appears in the c3h4 / superminimal
mechanisms (RMG-Py/examples/rmg/{c3h4,superminimal}/input.py use the
'default' family set; the species are CH2, C2H2, H2, O2, N2) plus the
RMG-Py test fixtures (the family set of RMG-Py/tests/.../testing_database
that step-02/03 recorded against). The case set below is the UNION of the
step-02 recorded cases and those mechanism families:

    H_Abstraction, Disproportionation, R_Recombination,
    Birad_recombination, R_Addition_MultipleBond, 1,2_shiftC,
    Intra_ene_reaction, intra_H_migration, Singlet_Val6_to_triplet

Sources:
  - 'default': the family's definition is loaded from the REAL
    RMG-database input/kinetics/families via rmgpu's family loader
    (rmpgu/core/family.py), cross-checked against rmgdb.
  - 'test': the family exists only in RMG-Py's testing_database
    (1,2_shiftC, Singlet_Val6_to_triplet); the gate loads its groups.py
    directly (no rmgdb cross-check - those families are not in rmgdb).

Each case: (family, source, [reactant SMILES]) - unimolecular /
bimolecular gas-phase only (no surface / termolecular), matching the
step-02 reference's scope.
"""

CASES = [
    # --- H abstraction (real family; multiple H sites, resonance) ---
    ('H_Abstraction', 'default', ['C', '[H]']),
    ('H_Abstraction', 'default', ['CC', '[OH]']),
    ('H_Abstraction', 'default', ['CC(=O)C', '[H]']),
    ('H_Abstraction', 'default', ['C[CH]C', '[H]']),
    ('H_Abstraction', 'default', ['C1=CC=CC=C1', '[H]']),
    ('H_Abstraction', 'default', ['C=C', '[CH3]']),
    ('H_Abstraction', 'default', ['C(C)(C)C', '[CH3]']),
    ('H_Abstraction', 'default', ['CC(=O)[O]', '[H]']),
    ('H_Abstraction', 'default', ['C1CCC1', '[OH]']),
    ('H_Abstraction', 'default', ['C1CC1', '[CH3]']),
    ('H_Abstraction', 'default', ['CCO', '[OH]']),
    ('H_Abstraction', 'default', ['C#C', '[H]']),
    ('H_Abstraction', 'default', ['CCN', '[H]']),
    ('H_Abstraction', 'default', ['CC#C', '[CH3]']),
    # --- Disproportionation (real family; single-group split Root, 4 labeled
    #     atoms across 2 reactants) ---
    ('Disproportionation', 'default', ['CC[CH]', '[OH]']),
    ('Disproportionation', 'default', ['CC[CH]', '[CH3]']),
    # --- radical-radical recombination (real family; Root split,
    #     own_reverse=False -> no reverse-direction generation in the same
    #     family; the reverse family Bond_Dissociation is not in 'default') ---
    ('R_Recombination', 'default', ['[OH]', '[OH]']),
    ('R_Recombination', 'default', ['[CH3]', '[OH]']),
    ('R_Recombination', 'default', ['[CH3]', '[CH3]']),
    # --- multiple-bond addition (real family; double + triple + aromatic) ---
    ('R_Addition_MultipleBond', 'default', ['C=C', '[CH3]']),
    ('R_Addition_MultipleBond', 'default', ['C=C', '[OH]']),
    ('R_Addition_MultipleBond', 'default', ['C1=CC=CC=C1', '[H]']),
    ('R_Addition_MultipleBond', 'default', ['C=O', '[CH3]']),
    ('R_Addition_MultipleBond', 'default', ['C#C', '[H]']),
    ('R_Addition_MultipleBond', 'default', ['C#C', '[CH3]']),
    # --- intra families (real, self-reverse, label relabeling) ---
    ('intra_H_migration', 'default', ['C[CH]CCC']),
    ('intra_H_migration', 'default', ['C[CH]C1CCCCC1']),
    ('intra_H_migration', 'default', ['C(C)(C)[CH]C(C)(C)C']),
    ('Intra_ene_reaction', 'default', ['C[CH]C1=CC=CC=C1']),
    # --- 1,2 shifts (test family; self-reverse) ---
    ('1,2_shiftC', 'test', ['CC[CH]C']),
    ('1,2_shiftC', 'test', ['CCC[CH]C(C)C']),
    # --- O2 dissociation (test; irreversible, own_reverse=False) ---
    ('Singlet_Val6_to_triplet', 'test', ['O=O']),
]

# The families this case set covers (the gate also loads the whole 'default'
# set to record the blocked-families list for job 06).
FAMILIES = sorted({f for f, _s, _r in CASES})

# Timing gate: product enumeration for the 10-atom-class reactant < 5 s.
# The gate records the per-case enumeration wall time and asserts the
# MAXIMUM across the case set stays under this floor (the set's largest
# reactants exceed 10 atoms: intra_H_migration C(C)(C)[CH]C(C)(C)C is
# C8H17 = 25 atoms explicit, 1,2_shiftC CCC[CH]C(C)C is C7H13 = 20; the
# floor is therefore stronger than the brief's 10-atom requirement).
TIMING_SECONDS = 5.0
