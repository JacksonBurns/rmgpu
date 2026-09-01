#!/usr/bin/env python3
"""Settle the atom-type tree relation: is `specific`/`generic` transitive?
This determines how AtomType.is_specific_case_of must be ported."""
from rmgpy.molecule.atomtype import ATOMTYPES

for probe in ['C', 'R!H', 'R', 'Ca', 'Cd', 'Cs', 'H', 'H0', 'O', 'O2s']:
    at = ATOMTYPES[probe]
    print('%-5s generic=%s' % (probe, [g.label for g in at.generic]))
    print('%-5s specific(n=%d)=%s' % (probe, len(at.specific),
                                      [s.label for s in at.specific]))
    print('---')
# Is 'Ca' a direct child of 'C'? Is 'Ca' in 'R!H'.specific?
print("Ca in C.specific:", any(s.label == 'Ca' for s in ATOMTYPES['C'].specific))
print("Ca in R!H.specific:", any(s.label == 'Ca' for s in ATOMTYPES['R!H'].specific))
print("Ca in R.specific:", any(s.label == 'Ca' for s in ATOMTYPES['R'].specific))
print("Ca.generic:", [g.label for g in ATOMTYPES['Ca'].generic])
# test the is_specific_case_of predicate directly
ca = ATOMTYPES['Ca']
print("Ca.is_specific_case_of(C):", ca.is_specific_case_of(ATOMTYPES['C']))
print("Ca.is_specific_case_of(R!H):", ca.is_specific_case_of(ATOMTYPES['R!H']))
print("Ca.is_specific_case_of(R):", ca.is_specific_case_of(ATOMTYPES['R']))
print("C.is_specific_case_of(Ca):", ATOMTYPES['C'].is_specific_case_of(ca))
