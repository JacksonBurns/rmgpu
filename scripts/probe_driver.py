"""Probe the draft pdep driver end-to-end on the 2-isomer reference."""
import json
import os
import tempfile
import numpy as np

from rmgpu.pdep.network import Network
from rmgpu.pdep import driver


def main():
    d = json.load(open("gates/baselines/job07/toy_2isomer_ref.json"))
    nb = d["network"]
    gp = d["grain_params"]
    Tlist = np.array(d["kTp"]["Tlist"])
    Plist = np.array(d["kTp"]["Plist"])
    K_ref = np.array(d["kTp"]["K_ref"])
    snapshots = d["snapshots"]

    # Build the rmgpu Network (2 isomers, no reactant channels, 1 product channel)
    net = Network(
        label="toy-2isomer",
        n_isom=nb["n_isom"], n_reac=nb["n_reac"], n_prod=nb["n_prod"],
        E0=np.array(nb["E0_isomers"] + [nb["E0_product"]]),
        Tmin=gp["Tmin"], Tmax=gp["Tmax"], Pmin=gp["Pmin"], Pmax=gp["Pmax"],
        grain_size=gp["max_grain_size"], grain_count=gp["min_grain_count"],
    )

    # state_provider seam: T (float) -> recorded per-T state dict
    def provider(T):
        return snapshots[str(round(float(T), 6))]
    net.state_provider = provider

    # The driver's network view + the net reactions (from the reference)
    pdp = driver.PDepNetwork(label="toy-2isomer", network=net)
    for nr in d["net_reactions"]:
        pdp.reactions.append(
            driver.PDepReaction(
                label="cfg%d>cfg%d" % (nr["from_cfg"], nr["to_cfg"]),
                from_cfg=nr["from_cfg"], to_cfg=nr["to_cfg"],
                n_reactants=nr["n_reactants"],
            ))

    tmp = tempfile.mkdtemp()
    res = driver.run_pdep(
        pdp, block=None, method="cse",
        interpolation_model=("PDepArrhenius",),
        Tlist=Tlist, Plist=Plist, output_dir=tmp, error_check=True,
    )
    print("== run_pdep OK ==")
    print("n_fitted", res.n_fitted, "n_reactions", res.n_reactions)
    print("wall_s", round(res.wall_s, 3), "wall_solve_s", round(res.wall_solve_s, 3))
    print("yaml_path", res.yaml_path)
    K = res.K
    # Compare the driver's K (from the network solve) vs RMG's K_ref
    worst = 0.0
    for t in range(len(Tlist)):
        for p in range(len(Plist)):
            dlt = np.abs(K[t, p] - K_ref[t, p])
            rel = dlt / np.maximum(np.abs(K_ref[t, p]), 1e-300)
            rel = rel[np.isfinite(rel)]
            if len(rel):
                worst = max(worst, float(rel.max()))
    print("K (network solve) vs K_ref max rel diff: %.4e" % worst)

    for rxn in res.reactions:
        kin = rxn.kinetics
        if kin is None:
            print("  %s: kinetics=None (zero channel)" % rxn.label)
            continue
        print("  %s: fitted %s log_rms=%.3f" % (rxn.label, type(kin).__name__, rxn.fit_log_rms))
        # check the fit reproduces the grid for this net reaction
        kdata = K[:, :, rxn.to_cfg, rxn.from_cfg]
        if (kdata > 0).all():
            w = 0.0
            for t in range(len(Tlist)):
                for p in range(len(Plist)):
                    km = kin.get_rate_coefficient(Tlist[t], Plist[p])
                    kd = kdata[t, p]
                    if kd > 0:
                        w = max(w, abs(km - kd) / kd)
            print("      fit max rel diff vs grid: %.4e" % w)

    # yaml round-trip
    doc = driver.load_network_yaml(res.yaml_path)
    print("yaml round-trip: network=%s n_isom=%s n_rxn=%d K_grid shape=%s" %
          (doc["network"], doc["n_isom"], len(doc["reactions"]),
           np.array(doc["K_grid"]).shape if doc["K_grid"] else None))


if __name__ == "__main__":
    main()
