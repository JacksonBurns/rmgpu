import logging
from pathlib import Path
import lightning.pytorch as pl
import pandas as pd
import torch

from chemprop import data
from config import KINETICS_TARGETS, THERMO_TARGETS
from models import CHEMELEON_MOL_FEATURIZER, RIGR_RXN_FEATURIZER
from chemprop.models.utils import load_model as chemprop_load_model

import logging
from contextlib import contextmanager

import logging
import warnings
from contextlib import contextmanager

@contextmanager
def suppress_lightning_logs(level=logging.WARNING):
    # Loggers to suppress
    logger_names = [
        "lightning.pytorch.utilities.rank_zero",
        "lightning.pytorch.accelerators.cuda",
    ]
    loggers = [logging.getLogger(name) for name in logger_names]
    original_levels = [logger.level for logger in loggers]

    with warnings.catch_warnings():
        # Suppress LeafSpec deprecation warning
        warnings.filterwarnings(
            "ignore",
            message=r".*isinstance\(treespec, LeafSpec\).*",
        )
        # Suppress the DataLoader num_workers bottleneck warning
        warnings.filterwarnings(
            "ignore",
            message=r".*does not have many workers which may be a bottleneck.*",
        )

        try:
            for logger in loggers:
                logger.setLevel(level)
            yield
        finally:
            for logger, original_level in zip(loggers, original_levels):
                logger.setLevel(original_level)


class CheMeleonThermoPredictor:
    def __init__(self, ckpt_path: str | Path = Path("chemeleon_thermo_122e91.ckpt"), batch_size: int = 64):
        self.model = chemprop_load_model(ckpt_path)
        self.model.eval()
        self.batch_size = batch_size

    def __call__(self, smiles: list[str]) -> pd.DataFrame:
        datapoints = [
            data.MoleculeDatapoint.from_smi(smi, keep_h=True, add_h=True)
            for smi in smiles
        ]
        dset = data.MoleculeDataset(datapoints, CHEMELEON_MOL_FEATURIZER)
        loader = data.build_dataloader(dset, batch_size=self.batch_size, shuffle=False)

        with suppress_lightning_logs():
            trainer = pl.Trainer(accelerator="auto", devices=1, logger=False, enable_progress_bar=False)
            raw_preds = trainer.predict(self.model, loader)
        preds = torch.cat(raw_preds).detach().cpu().numpy()

        df = pd.DataFrame(preds, columns=list(THERMO_TARGETS))
        df.insert(0, "smiles", smiles)
        return df

class ChempropKineticsPredictor:
    def __init__(self, ckpt_path: str | Path = Path("chemprop_kinetics_122e91.ckpt"), batch_size: int = 64):
        self.model = chemprop_load_model(ckpt_path)
        self.model.eval()
        self.batch_size = batch_size

    def __call__(self, rxn_smiles: list[str]) -> pd.DataFrame:
        datapoints = [
            data.ReactionDatapoint.from_smi(smi, keep_h=True, add_h=True)
            for smi in rxn_smiles
        ]
        dset = data.ReactionDataset(datapoints, RIGR_RXN_FEATURIZER)
        loader = data.build_dataloader(dset, batch_size=self.batch_size, shuffle=False)

        with suppress_lightning_logs():
            trainer = pl.Trainer(accelerator="auto", devices=1, logger=False, enable_progress_bar=False)
            raw_preds = trainer.predict(self.model, loader)
        preds = torch.cat(raw_preds).detach().cpu().numpy()

        df = pd.DataFrame(preds, columns=list(KINETICS_TARGETS))
        df.insert(0, "rxn_smiles", rxn_smiles)
        return df

if __name__ == "__main__":
    predictor_thermo = CheMeleonThermoPredictor()
    predictor_kinetics = ChempropKineticsPredictor()
    print(predictor_thermo(["C", "CC", "CCC"]))
    print(predictor_kinetics(["[O:1]([C:2]([C:3]([C:4](=[O:5])[C:6]([O:7][H:15])([H:13])[H:14])([H:11])[H:12])([H:9])[H:10])[H:8]>>[C:3](=[C:4]=[O:5])([H:11])[H:12].[C:6]([O:7][H:15])([H:8])([H:13])[H:14].[O:1]=[C:2]([H:9])[H:10]", "[O:1]=[c:2]1[n:3]([H:7])[c:4]([H:8])[n:5][o:6]1>>[N:3]([C:4]#[N:5])([H:7])[H:8].[O:1]=[C:2]=[O:6]"]))
