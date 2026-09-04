# job-06/step-08-sparse-dae-ml-batch

Job: job-06 - Core/edge loop + torchdae reactor
Prereq: job-06/step-07-fix-gate completed, gate re-opened, core/edge loop with promote/demote in place, but c3h4 enlarge is OOM/slow due to dense DAE tensors and per-reaction ML estimator calls.

Goal
----
Make the core/edge loop actually runnable on CPU for c3h4 by:
1. Replacing dense stoichiometric tensors in `rmgpu/reactor/simulator.py` with sparse-aware DAE integration. The stoichiometric matrix is ~99% sparse for large mechanisms. Use per-reaction species index lists and sparse scatter-add for `dydt`. Keep torchdae TR-BDF2 backend, but avoid materializing (n_rx x n_sp) dense tensors.
2. Add caching + batching to `rmgpu/ml/kinetics_estimator.py` to avoid per-reaction Python overhead and overflow warnings. Cache predictions by reaction SMILES, batch `predict_raw` calls with batch_size >= 128.

Deliverables
------------
- rmgpu/reactor/simulator.py
  * Refactor `simulate_mole_fractions` to build per-reaction species index lists from `nu` and use sparse gathers/scatters for rates and dydt. Keep FP32, keep dilution term.
  * No dense `neg_nu` / `pos_nu` tensors of size (n_sp, n_rx). Replace with lists of indices/values.
- rmgpu/ml/kinetics_estimator.py
  * Add `self._cache = {}` in `__init__`
  * Add `predict_batch(smiles_list)` that batches `predict_raw` with batch_size=256
  * Modify `predict` to check cache first, then fallback to batching via a simple pending queue or direct batch of 1 with cache.
  * Optional: add a small LRU size limit to avoid unbounded growth.
- tests/test_reactor_sparse.py
  * Minimal sanity: simulate a 3-species 2-reaction system with sparse `nu` and verify mole-fraction sum conserved to 1e-9.
- tests/test_ml_batch_cache.py
  * Verify cache hit avoids second model call, and batch prediction returns same values as single predictions within tolerance.

Checks
------
pytest tests/test_reactor_sparse.py -q
pytest tests/test_ml_batch_cache.py -q
Run a short c3h4 dry-run with max_iterations=1 to ensure enlarge + simulate completes without OOM and finishes in < 5 min on CPU.

Done protocol
-------------
1. Commit code: git commit -am "job-06/step-08: sparse DAE + ML batch/cache"
2. STATUS.md: set step row to done, append session log entry with built/checks/commits/next
3. Update NEXT pointer to prompts/steps/job-06-step-09-... (or next step in job-06)
4. Write reports/job-06-step-08-sparse-dae-ml-batch.md with what was built, checks, deviations, next step notes.
5. STOP. Do not start next step.

Notes
-----
* Keep FP32 as set in previous step.
* Do not change the gate logic; only make it runnable.
* If batching requires refactor of `KineticsML.predict` signature, keep backward compatibility: `predict(reaction_smiles, degeneracy)` should still work, internally using cache/batch.
