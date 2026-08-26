# job-01/step-03 report

Built:
- rmgpu/molecule/adjlist.py: parse_adjlist() parses RMG adjacency lists into atom and bond metadata; serialize_adjlist() renders them as deterministic adjacency-list text. Added element validation and sorted bond output.
- tests/test_roundtrip_check.py: six round-trip and string-stability checks for ethane, methane, water, ethylene, benzene, and a radical.
- rmgpu/molecule/molecule.py: fixed from_adjacency_list() to add explicit hydrogens before creating the final molecule, and fixed is_isomorph() to compare molecules with explicit hydrogens.

Checks:
- pytest tests/test_adjlist.py tests/test_roundtrip_check.py -v
- Result: 22 passed.
- Round-trip stability verified for six molecules; serialization output was stable after each parse-serialize cycle.

Reference reads beyond the step list: none.

Deviations: the requested examples/rmg/ directory was absent, so the prescribed species-specific round-trip was covered by an equivalent set of representative molecules. No external code was copied.

Next step: begin atom-type assignment and compare results against the RMG-Py atom-type baseline.