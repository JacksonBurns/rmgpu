# job-01/step-08: Fix parity failures from job-01 gate

Job: job-01 - Units + molecule layer
Prereq: job-01/step-07 gate is RED
This step fixes the three RED checks from the gate before job-01 can close.

## Goal

Fix the parity failures found by gate_01.py:
1. adjlist_parity: rmgpu adjlist strings must match RMG-Py formatting
2. atomtype_parity: rmgpu atom types must match RMG-Py assignments
3. symmetry_parity: rmgpu symmetry numbers must match RMG-Py

## Reference to read

- Read RMG-Py adjlist.py to_adjacency_list() for exact formatting
- Read RMG-Py atomtype.py for atom type assignment logic
- Read RMG-Py symmetry.py for symmetry number calculation

## Checks (must run and pass before claiming done)

```
/home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/ -q
/home/jackson/miniforge3/envs/rmgpu/bin/python gates/gate_01.py
```

All checks must be GREEN, or REDs must be recorded precisely.

## Pitfalls

- RMG-Py adjlist formatting has specific column widths and ordering rules
- Atom types in RMG-Py use specific labels like C5s, O2s, etc.
- Symmetry number calculation in RMG-Py is complex; do not simplify
