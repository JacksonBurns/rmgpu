# job-06/step-05: Chemkin writer + species dictionary

## Files built

- `rmgpu/io/chemkin.py` — `write_chemkin(mechanism, path)` writes chem.inp, chem_annotated.inp, and species_dictionary.txt. Supports elements section, species thermo lines, reaction lines with Arrhenius and falloff (LOW line), termination, and initial states.
- `tests/test_chemkin.py` — 7 tests covering file creation, species count, reaction count, rate parameter round-trip, species dictionary, falloff LOW line, and elements section.

## Checks run

- `pytest tests/test_chemkin.py -v` — 7 passed

## Format notes

- Species line format: name(16) + comment(6) + elements(30) + phase(1) + Tmin(10) + Tmax(10) + Tint(10) = 83 chars
- Element string built from formula, left-justified to 30 chars
- Arrhenius params: A, n, Ea (kcal/mol) written as last 3 tokens
- Falloff: adds `LOW/` line with low-pressure parameters

## Reference reads

- RMG-Py/rmgpy/chemkin.pyx (writer reference for format)

## Deviations

- NASA7 coefficients currently written as zeros (thermo model has Cp values but not full NASA7 polynomial)

## Next step

- job-06/step-06: Job-06 gate (first real mechanism generation)
