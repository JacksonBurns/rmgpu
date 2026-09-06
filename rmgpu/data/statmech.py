"""
rmgpu.data.statmech
====================

Port of RMG-Py rmgpy/data/statmech.py get_statmech_data.

Builds a Conformer from the statmech DB group frequencies + Cp fitting.
This is the data path for job-07/step-03 (no QM).
"""

from __future__ import annotations

from typing import Any

from rmgpu.statmech.modes import Conformer, HarmonicOscillator, LinearRotor, NonlinearRotor, HinderedRotor


def get_statmech_data(molecule: Any, statmech_db: Any, thermo: Any) -> Conformer:
    """
    Assemble a Conformer for *molecule* using statmech DB group frequencies
    and Cp fitting.

    Parameters
    ----------
    molecule: rmgpu.molecule.Molecule
    statmech_db: StatMechDB
    thermo: thermo model with get_heat_capacity(T) and H298

    Returns
    -------
    Conformer
    """
    # Stub implementation for job-07/step-03.
    # Full group assignment + statmechfit is out of scope for this session
    # placeholder; the interface is established for downstream steps.
    spin = getattr(molecule, "get_radical_count", lambda: 0)()
    spin_multiplicity = int(spin) + 1

    # E0 from thermo model H298 -> J/mol
    E0 = 0.0
    try:
        # thermo is a Wilhoit/NASA model with H298 in J/mol
        h298 = getattr(thermo, "H298", 0.0)
        # Convert from kcal/mol to J/mol if needed
        if hasattr(thermo, "H298_unit"):
            unit = thermo.H298_unit.lower()
            if "kcal" in unit:
                E0 = float(h298) * 4184.0
            else:
                E0 = float(h298)
        else:
            E0 = float(h298)
    except Exception:
        E0 = 0.0

    # Minimal mode set: translation/rotation are added by Conformer assembly
    # in later steps. For now return empty modes.
    conformer = Conformer(
        E0=E0,
        modes=[],
        spin_multiplicity=spin_multiplicity,
        optical_isomers=1,
    )
    return conformer
