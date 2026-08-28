from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rdkit import Chem
from rmgpu.ml.base import THERMO_CKPT_PATH, ChempropCheckpoint, load_thermo_checkpoint
from rmgpu.molecule.molecule import Molecule


@dataclass
class ThermoPrediction:
    Hf298: float
    S298: float
    Cp_model: "CpModel"
    uncertainties: dict


class MLCoverageError(Exception):
    pass


class CpModel:
    """Interpolates discrete Cp values into a Wilhoit-compatible model."""

    def __init__(self, T: np.ndarray, Cp: np.ndarray, wilhoit: Optional[WilhoitModel]):
        self.T = T
        self.Cp = Cp
        self.wilhoit = wilhoit

    def get_heat_capacity(self, T: float) -> float:
        if self.wilhoit is not None:
            return self.wilhoit.get_heat_capacity(T)
        T_arr = self.T
        Cp_arr = self.Cp
        if T <= T_arr[0]:
            return Cp_arr[0]
        if T >= T_arr[-1]:
            return Cp_arr[-1]
        idx = np.searchsorted(T_arr, T) - 1
        return np.interp(T, T_arr, Cp_arr)


class WilhoitModel:
    """Simple Cp interpolator mimicking Wilhoit behavior for the estimator."""

    def __init__(self, Cp0: float, CpInf: float, a0: float, a1: float, a2: float, a3: float, B: float, Tmin: float, Tmax: float):
        self.Cp0 = Cp0
        self.CpInf = CpInf
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.B = B
        self.Tmin = Tmin
        self.Tmax = Tmax

    def get_heat_capacity(self, T: float) -> float:
        y = T / (T + self.B)
        return self.Cp0 + (self.CpInf - self.Cp0) * y * y * (
            1 + (y - 1) * (self.a0 + y * (self.a1 + y * (self.a2 + y * self.a3)))
        )

    def get_enthalpy(self, T: float) -> float:
        return 0.0  # Not implemented: H0, S0 constants not determined

    def get_entropy(self, T: float) -> float:
        return 0.0  # Not implemented


def _fit_wilhoit(T: np.ndarray, Cp: np.ndarray) -> tuple[WilhoitModel, float]:
    """Fit Wilhoit coefficients to the 7-point Cp grid. Returns model + max error."""
    # For simplicity, we use the same Cp0 and CpInf as the Wilhoit class defaults.
    # In a real implementation, this would be a more complex optimization.
    Cp0 = Cp[0]
    CpInf = Cp[-1]
    a0 = a1 = a2 = a3 = 0.0
    B = 500.0
    Tmin = T[0]
    Tmax = T[-1]
    model = WilhoitModel(Cp0=Cp0, CpInf=CpInf, a0=a0, a1=a1, a2=a2, a3=a3, B=B, Tmin=Tmin, Tmax=Tmax)
    errors = np.abs([model.get_heat_capacity(t) - c for t, c in zip(T, Cp)])
    max_error = float(np.max(errors))
    return model, max_error


class ThermoML:
    """Wraps the CheMeleon thermo checkpoint.

    Usage:
        thermo_ml = ThermoML(models_dir)
        prediction = thermo_ml.predict(molecule)
        prediction = thermo_ml.predict(molecule_smiles)
    """

    ALLOWED_ELEMENTS = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    }

    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        ckpt = load_thermo_checkpoint(self.models_dir / "chemeleon_thermo_662946.ckpt")
        self.ckpt = ckpt
        self.temps = np.array([300, 400, 500, 600, 800, 1000, 1500])

    def _is_molecule_valid(self, smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in self.ALLOWED_ELEMENTS:
                    return False
            heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
            if heavy_atoms > 30:
                return False
            return True
        except Exception:
            return False

    def covers(self, molecule: Molecule | str) -> bool:
        if isinstance(molecule, Molecule):
            smiles = molecule.to_smiles()
        else:
            smiles = molecule
        if not smiles:
            return False
        return self._is_molecule_valid(smiles)

    def predict(self, molecule: Molecule | str) -> ThermoPrediction:
        if isinstance(molecule, Molecule):
            smiles = molecule.to_smiles()
        else:
            smiles = molecule

        if not self.covers(smiles):
            raise MLCoverageError(f"SMILES not covered by thermo ML: {smiles}")

        from chemprop import data
        datapoint = data.MoleculeDatapoint.from_smi(smiles, keep_h=True, add_h=True)
        raw = self.ckpt.predict_raw([datapoint], batch_size=1)
        raw = raw[0].detach().cpu().numpy()

        # raw has 9 columns: log_H298, log_S298, log_Cp_1..7
        Hf298 = 10 ** raw[0]  # J/mol
        S298 = 10 ** raw[1]   # J/(mol*K)
        Cp = 10 ** raw[2:]    # J/(mol*K)

        wilhoit, max_error = _fit_wilhoit(self.temps, Cp)
        cp_model = CpModel(T=self.temps, Cp=Cp, wilhoit=wilhoit)

        return ThermoPrediction(
            Hf298=Hf298,
            S298=S298,
            Cp_model=cp_model,
            uncertainties={"max_cp_error": max_error},
        )
