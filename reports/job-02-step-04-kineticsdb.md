# job-02/step-04: KineticsDB Facade + Family Definition Storage

## What was built

### Files
1. **rmgpu/db/loaders.py** - Extended `KineticsDB` facade with:
   - `get_library_names()` - returns all kinetics library names
   - `get_library_reaction_count(library_name)` - returns reaction count for a specific library
   - `get_family_names()` - returns all family names
   - `get_family_definition(name)` - returns parsed family definition with template and recipe
   - `get_family_groups(family_name)` - returns all groups for a family
   - `get_reaction_by_reaction(reaction)` - substructure match lookup for reactions

2. **rmgpu/data/kinetics.py** - Updated kinetics retrieval logic:
   - `lookup_kinetics()` - looks up reaction kinetics via substructure match, falls back to ML estimator (TODO: job-04)

3. **tests/test_kineticsdb.py** - 14 tests covering:
   - Basic counts (libraries, reactions, families)
   - Library and family lookups
   - Reaction substructure matching
   - Family definition parsing
   - Family storage investigation

## Family Definition Storage Finding

**YES - rmgdb stores family definitions.** Specifically:

- **Family count**: 129 families in `kinetics_families_table`
- **Family groups**: 8502 groups in `kinetics_family_groups_table`
- **Family rules**: 0 rules in `kinetics_family_rules_table` (empty table)

### What's stored:
- Family templates (stored as string representations, parsed via `ast.literal_eval`)
- Family recipes (stored as string representations, parsed via `ast.literal_eval`)
- Family groups with adjacency lists
- Family metadata (reversible, reverse_map, reactant_num, product_num, etc.)

### What's missing:
- Family rules are empty - this is expected per the plan (job-05 will implement rate rules)
- Training reactions are empty (the "training depository" is just a library in rmgpu, feeding rate-rule TRAINING which is a deleted feature)

### Job-05 Strategy:
Since family definitions are already stored in rmgdb with templates and recipes, job-05 can:
1. Parse the stored templates and recipes using `ast.literal_eval`
2. Implement the recipe engine to apply these recipes to reactions
3. Implement rate rules (the empty `kinetics_family_rules_table` will be populated)

## Checks

All checks passed:
```bash
$ /home/jackson/miniforge3/envs/rmgpu/bin/python -m pytest tests/test_kineticsdb.py -q
..............                                                           [100%]
14 passed in 3.36s
```

## Reference reads beyond the list
- None

## Deviations
- None

## What the next step should know first
- KineticsDB facade is complete with all necessary lookup methods
- Family definitions are stored in rmgdb with templates and recipes
- Job-05 can directly consume family definitions from rmgdb
- The training depository is just a library (not a separate training data path)