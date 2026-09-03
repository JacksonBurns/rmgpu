#!/usr/bin/env python3
"""Probe: does RMG-Py generate the long O-chains (O_n, H-O_n-H, O_n diradicals)
that rmgpu is over-promoting into the superminimal core?

Run in rmg_env:
    /home/jackson/miniforge3/envs/rmg_env/bin/python scripts/_probe_o_chain.py

Method: load the 5 superminimal families + primaryThermoLibrary exactly as the
superminimal example does, then feed the O-species pairs that the O-chain
reaction family (Birad_recombination, H_Abstraction, Disproportionation)
would act on, and print the products RMG-Py generates. If RMG-Py generates the
same long O-chains, the divergence is a SCREENING problem (rmgpu promotes them
to core, RMG-Py prunes them). If RMG-Py does NOT generate them, the divergence
is an ENUMERATION/forbidden-structure problem.
"""
import copy
import os
import sys


def main():
    from rmgpy.data.rmg import RMGDatabase

    DB = "/home/jackson/rmgpu/RMG-database/input"
    db = RMGDatabase()
    db.load(path=DB,
            thermo_libraries=["primaryThermoLibrary"],
            reaction_libraries=[],
            seed_mechanisms=[],
            kinetics_families=["H_Abstraction", "Disproportionation",
                               "R_Recombination", "Birad_recombination",
                               "Birad_R_Recombination"],
            kinetics_depositories=["training"],
            depository=False, solvation=False, surface=False)
    fams = [db.kinetics.families[n] for n in
            ["H_Abstraction", "Disproportionation", "R_Recombination",
             "Birad_recombination", "Birad_R_Recombination"]]

    from rmgpy.molecule.molecule import Molecule

    def mol(adj):
        m = Molecule()
        m.from_adjacency_list(adj)
        return m

    SP = {
        "H": mol("1 H u1 p0 c0\n"),
        "H2": mol("1 H u0 p0 c0 {2,S}\n2 H u0 p0 c0 {1,S}\n"),
        "O": mol("1 O u2 p2 c0\n"),
        "O2": mol("multiplicity 3\n1 O u1 p2 c0 {2,S}\n2 O u1 p2 c0 {1,S}\n"),
        "OH": mol("multiplicity 2\n1 O u1 p2 c0 {2,S}\n2 H u0 p0 c0 {1,S}\n"),
        "H2O": mol("1 O u0 p2 c0 {2,S} {3,S}\n2 H u0 p0 c0 {1,S}\n3 H u0 p0 c0 {1,S}\n"),
        "HO2": mol("multiplicity 2\n1 O u1 p2 c0 {2,S}\n2 O u0 p2 c0 {1,S} {3,S}\n3 H u0 p0 c0 {2,S}\n"),
    }

    # The O-chain growth pairs: O2+O2, O+O2, O+OH, O+O, HO2+HO2, O2+OH ...
    pairs = [
        ("O2", "O2"), ("O", "O2"), ("O", "O"), ("O", "OH"), ("O", "H2O"),
        ("OH", "OH"), ("OH", "O2"), ("OH", "H2O"), ("OH", "O"), ("OH", "H2"),
        ("HO2", "HO2"), ("HO2", "O"), ("HO2", "OH"), ("HO2", "H"),
        ("O2", "OH"), ("O2", "H"), ("O", "H2"), ("O", "H"), ("H", "O2"),
        ("H", "OH"), ("H", "H"), ("H2", "OH"), ("H2", "O"),
    ]

    from rmgpy.molecule.molecule import Molecule as _M

    def formula(m):
        c = {}
        for a in m.atoms:
            c[a.symbol] = c.get(a.symbol, 0) + 1
        return "".join("%s%d" % (k, v) for k, v in sorted(c.items()))

    print("=== RMG-Py product enumeration on O-species pairs (5 superminimal fams) ===")
    seen = set()
    for fam in fams:
        hits = []
        for la, lb in pairs:
            a2, b2 = copy.deepcopy(SP[la]), copy.deepcopy(SP[lb])
            try:
                rxns = fam.generate_reactions([a2, b2])
            except Exception:
                rxns = []
            for r in rxns:
                ps = tuple(sorted(formula(m) for m in r.products))
                rk = (fam.label, la, lb, ps)
                if rk in seen:
                    continue
                seen.add(rk)
                hits.append("  %s  %s+%s -> %s" % (fam.label, la, lb, "+".join(ps)))
        if hits:
            print("\n--- %s ---" % fam.label)
            for h in sorted(hits):
                print(h)

    print("\n=== distinct product FORMULAS across all pairs (this is what can reach the model) ===")
    allforms = set()
    for fam in fams:
        for la, lb in pairs:
            a2, b2 = copy.deepcopy(SP[la]), copy.deepcopy(SP[lb])
            try:
                rxns = fam.generate_reactions([a2, b2])
            except Exception:
                rxns = []
            for r in rxns:
                for m in r.products:
                    allforms.add(formula(m))
    for f in sorted(allforms, key=lambda x: (len(x), x)):
        print(" ", f)

    # Now the KEY question: if we seed H2O2/OH/HOOH and let them recombine, do
    # we get O3, O4, H2O3...? Feed the longer species too.
    print("\n=== longer O-species pairs (does RMG-Py grow O3->O4->O5 ...?) ===")
    # build O3, H2O2, HO2, O4 from adjacency lists
    EXTRA = {
        "O3": mol("1 O u1 p1 c0 {2,D}\n2 O u0 p2 c0 {1,D} {3,S}\n3 O u1 p2 c0 {2,S}\n"),
        "H2O2": mol("1 O u1 p2 c0 {2,S} {4,S}\n2 O u1 p2 c0 {1,S} {3,S}\n3 H u0 p0 c0 {2,S}\n4 H u0 p0 c0 {1,S}\n"),
    }
    sp_all = dict(SP)
    sp_all.update(EXTRA)
    for la in ["H2O2", "O3", "HO2", "OH", "O", "O2"]:
        for lb in ["O", "OH", "O2", "H", "H2O", "H2O2"]:
            if la not in sp_all or lb not in sp_all:
                continue
            a2, b2 = copy.deepcopy(sp_all[la]), copy.deepcopy(sp_all[lb])
            for fam in fams:
                try:
                    rxns = fam.generate_reactions([a2, b2])
                except Exception:
                    rxns = []
                for r in rxns:
                    print("  %s  %s+%s -> %s" % (fam.label, la, lb,
                          "+".join(sorted(formula(m) for m in r.products))))


if __name__ == "__main__":
    main()
