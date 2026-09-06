"""Probe: run the superminimal loop a few iterations and dump what the
screening sees (profile progression + sp_ratio), to diagnose why nothing is
promoted to the core. CPU-only.
    CUDA_VISIBLE_DEVICES= /home/jackson/miniforge3/envs/rmgpu/bin/python scripts/probe_superminimal.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import rmgpu.main as M  # noqa: E402
from rmgpu.core.loop import CoreEdgeLoop  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    input_path = os.path.join(REPO, "examples", "superminimal.yaml")
    model_input = M.load_input(input_path)
    databases = M._build_databases(model_input)
    ml = M._build_ml()
    fam_sel = getattr(model_input, "database", None)
    fam_sel = fam_sel.kinetics_families if fam_sel is not None else "default"
    families = M._build_families(fam_sel)
    seed_species = M._build_seed_species(model_input)
    sm_sp, sm_rx, sm_sum = M._build_seed_mechanisms(model_input, databases)
    T, P, imf, t_term, conv = M._build_reactor(model_input)
    from rmgpu.core.loop import RunContext, LoopConfig
    mb = getattr(model_input, "model", None)
    ctx = RunContext(
        databases=databases, ml=ml, families=families,
        seed_species=seed_species, initial_mole_fractions=imf,
        temperature=T, pressure=P,
        config=LoopConfig(tolerance_move_to_core=0.001,
                          tolerance_keep_in_edge=0.0, max_iterations=3),
        thermo_libraries=list(db_block.thermo_libraries or [])
        if (db_block := getattr(model_input, "database", None)) else None,
        reaction_libraries=[], termination_time=t_term,
        termination_conversion=conv)
    ctx.seed_mechanisms_species = sm_sp
    ctx.seed_mechanisms_reactions = sm_rx
    ctx.seed_mechanisms_summaries = sm_sum

    loop = CoreEdgeLoop(ctx)
    # re-seed like run()
    for sp in ctx.seed_species:
        sp.is_seed = True
        loop.model.add_species_to_core(sp)
        loop.species_by_key[__import__("rmgpu.core.loop", fromlist=["canonical_key"]).canonical_key(sp.molecule)] = sp
        loop._thermo(sp, 0)

    import logging
    logging.disable(logging.CRITICAL)

    for it in range(1, 4):
        loop.model.iteration_num = it
        loop.enlarge(it)
        profiles, promote, demote = loop.simulate(it)
        sim = loop._last_sim
        if profiles is not None:
            ys = np.array(profiles.ys, float)
            times = np.array(profiles.times, float)
            keys = sim["keys"]
            print(f"\n=== iteration {it} ===")
            print(f"t_end={times[-1]:.6g} s  n_steps={len(times)}  n_species={len(keys)}")
            last = ys[-1]
            top = sorted(range(len(keys)), key=lambda i: -last[i])[:12]
            print("top species at t_end:",
                  [(keys[i], f"{last[i]:.3e}") for i in top])
            print("promote:", len(promote), promote[:12])
            print("demote:", len(demote))
            sr = getattr(loop, "_last_sp_ratio", {})
            top_sr = sorted(sr.items(), key=lambda kv: -kv[1])[:12]
            print("top sp_ratio:", [(k, f"{v:.3e}") for k, v in top_sr])
        for k in promote:
            loop._promote(k)
        for k in demote:
            loop._demote(k)
        loop.prune()
        print(f"-> core {len(loop.model.core.species)} spc / "
              f"{len(loop.model.core.reactions)} rxn, edge "
              f"{len(loop.model.edge.species)} spc / "
              f"{len(loop.model.edge.reactions)} rxn")


if __name__ == "__main__":
    main()
