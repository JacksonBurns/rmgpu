from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rmgpu.ml.base import KINETICS_CKPT_PATH, ChempropCheckpoint, load_kinetics_checkpoint
from rmgpu.ml.thermo_estimator import MLCoverageError, ThermoML


@dataclass
class KineticsPrediction:
    A: float  # cm^3/(mol*s) (CGS)
    n: float
    Ea: float  # J/mol
    uncertainties: dict


class KineticsML:
    """Wraps the Chemprop kinetics checkpoint.

    Usage:
        kinetics_ml = KineticsML(models_dir)
        prediction = kinetics_ml.predict(reaction_smiles, degeneracy)
    """

    def __init__(self, models_dir: Path | str | None = None):
        if models_dir is None:
            from rmgpu.ml.base import MODELS_DIR
            self.models_dir = MODELS_DIR
        else:
            self.models_dir = Path(models_dir)
        ckpt = load_kinetics_checkpoint(self.models_dir / "chemprop_kinetics_662946.ckpt")
        self.ckpt = ckpt
        self.thermo_ml = ThermoML(self.models_dir)

    def _parse_reaction_smiles(self, reaction_smiles: str) -> tuple[str, str]:
        """Split reaction SMILES into reactant and product parts."""
        if ">>" not in reaction_smiles:
            raise ValueError(f"Invalid reaction SMILES (no '>>'): {reaction_smiles}")
        reactants_str, products_str = reaction_smiles.split(">>", 1)
        return reactants_str, products_str

    def _remove_atom_mappings(self, smiles: str) -> str:
        """Remove atom mapping labels from SMILES."""
        return re.sub(r":\d+", "", smiles)

    def covers(self, reaction_smiles: str) -> bool:
        """Check if both reactants and products are covered by thermo ML."""
        try:
            reactants_str, products_str = self._parse_reaction_smiles(reaction_smiles)
        except ValueError:
            return False

        reactants_smiles = self._remove_atom_mappings(reactants_str)
        products_smiles = self._remove_atom_mappings(products_str)

        # Check if reactants are covered
        if not self.thermo_ml.covers(reactants_smiles):
            return False

        # Check if products are covered
        if not self.thermo_ml.covers(products_smiles):
            return False

        return True

    def predict(self, reaction_smiles: str, degeneracy: float = 1.0) -> KineticsPrediction:
        """Predict Arrhenius parameters for a reaction.

        Args:
            reaction_smiles: Atom-mapped reaction SMILES (reactants>>products)
            degeneracy: Degeneracy factor for A conversion

        Returns:
            KineticsPrediction with A, n, Ea
        """
        if not self.covers(reaction_smiles):
            raise MLCoverageError(f"Reaction SMILES not covered by kinetics ML: {reaction_smiles}")

        from chemprop import data
        datapoint = data.ReactionDatapoint.from_smi(reaction_smiles, keep_h=True, add_h=True)
        raw = self.ckpt.predict_raw([datapoint], batch_size=1)
        raw = raw[0].detach().cpu().numpy()

        # raw has 3 columns: log10_A, n, Ea_J_mol
        log10_A = raw[0]  # log10 of per-site A (CGS cm^3/(mol*s))
        n = raw[1]  # linear
        Ea = raw[2]  # J/mol (linear, can be <= 0)

        # Boundary conversion (PLAN.md 3b): A = 10^pred_A * degeneracy
        A = 10 ** log10_A * degeneracy

        uncertainties = {}  # Add uncertainty info if needed

        return KineticsPrediction(
            A=A,
            n=float(n),
            Ea=float(Ea),
            uncertainties=uncertainties,
        )