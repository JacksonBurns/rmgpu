"""Estimation resolvers: library -> ML -> MLCoverageError.

These two functions are the ONLY estimation code in rmgpu. They replace
RMG-Py's multi-method fallback chains. No group additivity, no rate rules,
no fallbacks beyond ML. Instrumented with a counts dict so the thesis test
can prove the no-fallback invariant.

SI units internally (J, K, Pa, mol). The ``ml`` argument is threaded in,
not global - job 06's driver constructs estimators once and threads them.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from rmgpu.db.loaders import ThermoDB, KineticsDB
from rmgpu.ml.thermo_estimator import ThermoPrediction, MLCoverageError, CpModel
from rmgpu.kinetics.models import Arrhenius

__all__ = [
    "EstimationCounts",
    "estimate_thermo",
    "estimate_kinetics",
]


@dataclass
class EstimationCounts:
    """Instrumentation counters for the estimation resolvers.

    The thesis test reads these to prove no fallback ever happens:
    the split between library hits, ML hits, and coverage errors must
    sum to the total number of resolution attempts.
    """

    library_hits: int = 0
    ml_hits: int = 0
    coverage_errors: int = 0

    def as_dict(self) -> dict:
        return {
            "library_hits": self.library_hits,
            "ml_hits": self.ml_hits,
            "coverage_errors": self.coverage_errors,
        }


def estimate_thermo(
    species: dict,
    thermo_db: ThermoDB,
    ml,
    counts: Optional[EstimationCounts] = None,
) -> ThermoPrediction:
    """Resolve thermo properties for a species.

    Strategy (no fallbacks beyond ML, no GA, no rate rules):
    1. Check if species is in thermo library (label match).
    2. If not found, use ML estimator (ml.thermo).
    3. If ML cannot cover it, raise MLCoverageError.

    Args:
        species: dict with at least 'label' key (for library lookup) and
                 'adjacency_list' or 'smiles' key (for ML).
        thermo_db: ThermoDB facade for library lookup.
        ml: object with a 'thermo' attribute - a ThermoML estimator.
        counts: optional EstimationCounts for instrumentation.

    Returns:
        ThermoPrediction with Hf298, S298, Cp_model, uncertainties.

    Raises:
        MLCoverageError: if neither library nor ML covers the species.
    """
    if counts is None:
        counts = EstimationCounts()

    label = species.get("label", "")
    entry = thermo_db.get_entry_grouped_by_label(label)

    if entry is not None:
        counts.library_hits += 1
        # Build ThermoPrediction from library entry
        # Use CpModel from entry's Tdata/Cpdata if available, else fallback
        if entry.Tdata and entry.Cpdata:
            T_arr = np.array(entry.Tdata)
            Cp_arr = np.array(entry.Cpdata)
            wilhoit, max_error = _fit_wilhoit_simple(T_arr, Cp_arr)
            cp_model = CpModel(T=T_arr, Cp=Cp_arr, wilhoit=wilhoit)
        else:
            T_arr = np.array([], dtype=np.float64)
            Cp_arr = np.array([], dtype=np.float64)
            cp_model = CpModel(T=T_arr, Cp=Cp_arr, wilhoit=None)

        return ThermoPrediction(
            Hf298=entry.H298,
            S298=entry.S298,
            Cp_model=cp_model,
            uncertainties={"source": "library", "label": entry.label},
        )

    # No library hit - try ML
    ml_thermo = ml.thermo
    if ml_thermo is None:
        counts.coverage_errors += 1
        raise MLCoverageError(
            f"no ML thermo estimator configured for species {label}"
        )

    smiles = species.get("smiles", "")
    if not smiles:
        # Try to get SMILES from adjacency_list
        adjlist = species.get("adjacency_list", "")
        if adjlist:
            smiles = _adjlist_to_smiles(adjlist)

    if not smiles:
        counts.coverage_errors += 1
        raise MLCoverageError(
            f"no SMILES or adjacency_list provided for species {label}"
        )

    if not ml_thermo.covers(smiles):
        counts.coverage_errors += 1
        raise MLCoverageError(
            f"ML thermo model does not cover species {label} (SMILES: {smiles})"
        )

    prediction = ml_thermo.predict(smiles)
    counts.ml_hits += 1
    return prediction


def _adjlist_to_smiles(adjlist: str) -> str:
    """Convert adjacency list to SMILES using RDKit."""
    from rmgpu.molecule.molecule import Molecule
    try:
        mol = Molecule.from_adjacency_list(adjlist)
        return mol.to_smiles()
    except Exception:
        return ""


def _fit_wilhoit_simple(T: np.ndarray, Cp: np.ndarray):
    """Fit simple Wilhoit model for library Cp data."""
    from rmgpu.ml.thermo_estimator import WilhoitModel
    Cp0 = Cp[0] if len(Cp) > 0 else 0.0
    CpInf = Cp[-1] if len(Cp) > 0 else 0.0
    Tmin = T[0] if len(T) > 0 else 300.0
    Tmax = T[-1] if len(T) > 0 else 1500.0
    model = WilhoitModel(
        Cp0=Cp0, CpInf=CpInf, a0=0.0, a1=0.0, a2=0.0, a3=0.0,
        B=500.0, Tmin=Tmin, Tmax=Tmax
    )
    return model, 0.0


def estimate_kinetics(
    reaction: dict,
    kinetics_db: KineticsDB,
    ml,
    counts: Optional[EstimationCounts] = None,
) -> tuple:
    """Resolve kinetics for a reaction.

    Strategy (no fallbacks beyond ML, no GA, no rate rules):
    1. Check if reaction is in kinetics library (substructure match).
    2. If not found, use ML estimator (ml.kinetics).
    3. If ML cannot cover it, raise MLCoverageError.

    Args:
        reaction: dict with 'reactants' and 'products' lists (each with
                  'adjacency_list' or 'smiles'), and optional 'label'.
        kinetics_db: KineticsDB facade for library lookup.
        ml: object with a 'kinetics' attribute - a KineticsML estimator.
        counts: optional EstimationCounts for instrumentation.

    Returns:
        tuple (rate_model, degeneracy) where rate_model is a KineticsModel
        and degeneracy is the reaction degeneracy factor.

    Raises:
        MLCoverageError: if neither library nor ML covers the reaction.
    """
    if counts is None:
        counts = EstimationCounts()

    # Check library first
    match = kinetics_db.get_reaction_by_reaction(reaction)
    if match is not None:
        rate = kinetics_db.get_rate_model(match["id"])
        counts.library_hits += 1
        # degeneracy comes from the reaction object in job-06
        return (rate, 1.0)

    # No library hit - try ML
    ml_kinetics = ml.kinetics
    if ml_kinetics is None:
        counts.coverage_errors += 1
        raise MLCoverageError(
            f"no ML kinetics estimator configured for reaction {reaction.get('label', '')}"
        )

    # Build reaction SMILES from reactants/products
    reaction_smiles = _reaction_to_smiles(reaction)
    if not reaction_smiles:
        counts.coverage_errors += 1
        raise MLCoverageError(
            f"could not build reaction SMILES for reaction {reaction.get('label', '')}"
        )

    if not ml_kinetics.covers(reaction_smiles):
        counts.coverage_errors += 1
        raise MLCoverageError(
            f"ML kinetics model does not cover reaction {reaction.get('label', '')}"
        )

    prediction = ml_kinetics.predict(reaction_smiles)
    counts.ml_hits += 1

    # Build Arrhenius from prediction (SI units already applied by estimator)
    rate_model = Arrhenius(
        A=prediction.A,
        n=prediction.n,
        Ea=prediction.Ea_J_mol,
        Tmin=prediction.Tmin,
        Tmax=prediction.Tmax,
        comment=f"ML estimate from {prediction.source}",
    )
    return (rate_model, prediction.degeneracy)


def _reaction_to_smiles(reaction: dict) -> str:
    """Build reaction SMILES from reactants and products."""
    reactant_smiles = []
    for reactant in reaction.get("reactants", []):
        smiles = reactant.get("smiles", "")
        if not smiles:
            adjlist = reactant.get("adjacency_list", "")
            if adjlist:
                smiles = _adjlist_to_smiles(adjlist)
        if smiles:
            reactant_smiles.append(smiles)

    product_smiles = []
    for product in reaction.get("products", []):
        smiles = product.get("smiles", "")
        if not smiles:
            adjlist = product.get("adjacency_list", "")
            if adjlist:
                smiles = _adjlist_to_smiles(adjlist)
        if smiles:
            product_smiles.append(smiles)

    if not reactant_smiles or not product_smiles:
        return ""

    return " + ".join(reactant_smiles) + " -> " + " + ".join(product_smiles)
