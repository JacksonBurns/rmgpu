from pathlib import Path

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from config import THERMO_TARGETS, KINETICS_TARGETS
from data import (
    fetch_kinetics_training_data,
    fetch_thermo_training_data,
    df_to_chemprop,
)
from models import BoundedOutputTransform, get_thermo_model, get_kinetics_model


def fit(name: str, mpnn, train_loader, val_loader, test_loader):
    ckpt = ModelCheckpoint(
        Path(f"checkpoints/{name}"),
        "best-{epoch}-{val_loss:.2f}",
        "val_loss",
        mode="min",
    )
    es = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs", name=name, default_hp_metric=False),
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=100,
        callbacks=[ckpt, es],
    )

    trainer.fit(mpnn, train_loader, val_loader)
    results = trainer.test(
        mpnn, test_loader, weights_only=False, ckpt_path=ckpt.best_model_path
    )
    print(results)
    print("best checkpoint at:", ckpt.best_model_path)


def main():
    pl.seed_everything(42, workers=True)
    db_root = Path("/home/jackson/rmgpu/rmgdb/db")

    # thermo
    thermo_df = fetch_thermo_training_data(db_root / "thermo.db")
    thermo_df = thermo_df.replace([np.inf, -np.inf], np.nan)  # nan automatically masked in loss function
    # TODO: augment with resonance structures to make chemeleon resonance invariant (-ish)
    thermo_upper_bounds = thermo_df[list(THERMO_TARGETS)].max().values
    thermo_lower_bounds = thermo_df[list(THERMO_TARGETS)].min().values
    transform = BoundedOutputTransform(
        mean=thermo_df[list(THERMO_TARGETS)].mean().values,
        scale=thermo_df[list(THERMO_TARGETS)].std().values,
        lower_bounds=thermo_lower_bounds,
        upper_bounds=thermo_upper_bounds,
    )
    thermo_mpnn = get_thermo_model(transform)

    # kinetics
    kinetics_df = fetch_kinetics_training_data(db_root / "kinetics.db")
    kinetics_df = kinetics_df.replace([np.inf, -np.inf], np.nan)
    kinetics_upper_bounds = kinetics_df[list(KINETICS_TARGETS)].max().values
    kinetics_lower_bounds = kinetics_df[list(KINETICS_TARGETS)].min().values
    kinetics_transform = BoundedOutputTransform(
        mean=kinetics_df[list(KINETICS_TARGETS)].mean().values,
        scale=kinetics_df[list(KINETICS_TARGETS)].std().values,
        lower_bounds=kinetics_lower_bounds,
        upper_bounds=kinetics_upper_bounds,
    )
    kinetics_mpnn = get_kinetics_model(kinetics_transform)

    (
        thermo_train_loader,
        thermo_val_loader,
        thermo_test_loader,
        kinetics_train_loader,
        kinetics_val_loader,
        kinetics_test_loader,
    ) = df_to_chemprop(thermo_df, kinetics_df)

    fit(
        "thermo",
        thermo_mpnn,
        thermo_train_loader,
        thermo_val_loader,
        thermo_test_loader,
    )
    # fit(
    #     "kinetics",
    #     kinetics_mpnn,
    #     kinetics_train_loader,
    #     kinetics_val_loader,
    #     kinetics_test_loader,
    # )

if __name__ == "__main__":
    main()
