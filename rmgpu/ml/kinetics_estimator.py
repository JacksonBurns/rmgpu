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
        self._cache = {}
        self._pending = []  # list of (smiles, degeneracy) awaiting batch

    def _predict_batch(self, smiles_list):
        from chemprop import data
        datapoints = []
        for smi in smiles_list:
            datapoint = data.ReactionDatapoint.from_smi(smi, keep_h=True, add_h=True)
            datapoints.append(datapoint)
        batch_size = 256
        raw = self.ckpt.predict_raw(datapoints, batch_size=batch_size)
        # predict_raw returns a concatenated tensor or list of tensors
        if isinstance(raw, (list, tuple)):
            # Try to concat if tensors, else assume first element
            try:
                raw = torch.cat(raw)
            except Exception:
                raw = raw[0]
        raw = raw.detach().cpu().numpy()
        return raw

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
        key = (reaction_smiles, degeneracy)
        if key in self._cache:
            return self._cache[key]

        if not self.covers(reaction_smiles):
            raise MLCoverageError(f"Reaction SMILES not covered by kinetics ML: {reaction_smiles}")

        # Use batching for efficiency
        # Simple approach: if pending exists, batch them together
        # For now, just predict directly with cache
        raw_arr = self._predict_batch([reaction_smiles])
        raw = raw_arr[0]

        # raw has 3 columns: log10_A, n, Ea_J_mol
        log10_A = raw[0]  # log10 of per-site A (CGS cm^3/(mol*s))
        n = raw[1]  # linear
        Ea = raw[2]  # J/mol (linear, can be <= 0)

        # Boundary conversion (PLAN.md 3b): A = 10^pred_A * degeneracy
        A = 10 ** log10_A * degeneracy

        uncertainties = {}  # Add uncertainty info if needed

        pred = KineticsPrediction(
            A=A,
            n=float(n),
            Ea=float(Ea),
            uncertainties=uncertainties,
        )
        self._cache[key] = pred
        return pred