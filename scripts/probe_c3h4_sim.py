"""Probe: build the c3h4 seed-mechanism sim and measure, for increasing step
counts N, whether the profile is finite/physical, plus the wall time.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/probe_c3h4_sim.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/jackson/rmgpu/rmgpu")
import rmgpu.main as M  # noqa: E402
from rmgpu.core.loop import CoreEdgeLoop  # noqa: E402
from rmgpu.reactor.simulator import (  # noqa: E402
    RateParam, characteristic_rate, simulate_mole_fractions)
from rmgpu.core.loop import CHAR_RATE_TFACTOR  # noqa: E402


def main():
    import logging
    logging.disable(logging.CRITICAL)
    input_path = "/home/jackson/rmgpu/rmgpu/examples/c3h4.yaml"
    model_input = M.load_input(input_path)
    databases = M._build_databases(model_input)
    fam_sel = getattr(model_input, "database", None)
    families = M._build_families(fam_sel.kinetics_families)
    seed_species = M._build_seed_species(model_input)
    sm_sp, sm_rx, sm_sum = M._build_seed_mechanisms(model_input, databases)
    T, P, imf, t_term, conv = M._build_reactor(model_input)
    print(f"T={T} K P={P} Pa; seed: {len(sm_sp)} spc / {len(sm_rx)} rxn")

    # Build the same sim the loop would build for the seed core.
    from rmgpu.core.loop import RunContext, LoopConfig
    db_block = getattr(model_input, "database", None)
    ctx = RunContext(
        databases=databases, ml=M._build_ml(), families=families,
        seed_species=seed_species, initial_mole_fractions=imf,
        temperature=T, pressure=P, config=LoopConfig(),
        thermo_libraries=list(db_block.thermo_libraries or []),
        reaction_libraries=[], termination_time=t_term,
        termination_conversion=conv)
    loop = CoreEdgeLoop(ctx)
    for sp in ctx.seed_species:
        loop.model.add_species_to_core(sp)
    from rmgpu.core.loop import canonical_key
    for sp in sm_sp:
        k = canonical_key(sp.molecule)
        if k not in loop.species_by_key:
            loop.species_by_key[k] = sp
            loop.model.add_species_to_core(sp)
            loop._thermo(sp, 0)
    for rx in sm_rx:
        for s in rx.reactants + rx.products:
            k = canonical_key(s.molecule)
            if k not in loop.species_by_key:
                loop.species_by_key[k] = s
                loop.model.add_species_to_core(s)
        rk = [canonical_key(s.molecule) for s in rx.reactants]
        pk = [canonical_key(s.molecule) for s in rx.products]
        key = ".".join(sorted(rk)) + ">>" + ".".join(sorted(pk))
        loop.reactions[key] = {"rp": rx.rate_model, "react_keys": rk,
                               "prod_keys": pk, "family": "seed"}
    sim = loop._build_sim()
    keys, nu, rps, init = sim["keys"], sim["nu"], sim["rps"], sim["init"]
    char = characteristic_rate(nu, rps, T, P, init)
    t_end = CHAR_RATE_TFACTOR / char if char > 0 else 1.0
    t_end = min(t_end, float(t_term))
    print(f"n_species={len(keys)} n_rxn={len(rps)} char={char:.4e} "
          f"t_end(5/char)={t_end:.4e} s")

    for N in (32, 128, 512, 2048):
        h = t_end / N
        t0 = time.time()
        prof = simulate_mole_fractions(keys, nu, rps, T, P, init, t_end, h)
        dt = time.time() - t0
        ys = np.array(prof.ys, dtype=float)
        finite = np.all(np.isfinite(ys))
        sums = ys.sum(axis=1)
        in_range = np.all((ys >= -1e-6) & (ys <= 1 + 1e-6))
        max_dev = float(np.max(np.abs(sums - 1.0))) if finite else float("nan")
        print(f"N={N:5d} h={h:.3e}  finite={finite} in_range={in_range} "
              f"max|sum-1|={max_dev:.3e}  t={dt:.1f}s")


if __name__ == "__main__":
    main()
