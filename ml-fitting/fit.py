from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.seed import seed_everything

from config import THERMO_TARGETS, KINETICS_TARGETS
from data import (
    fetch_kinetics_training_data,
    fetch_thermo_training_data,
    df_to_chemprop,
)
from models import BoundedUnscaleTransform, get_thermo_model, get_kinetics_model


def fit(name: str, mpnn, train_loader, val_loader, test_loader):
    ckpt = ModelCheckpoint(
        Path(f"checkpoints/{name}"),
        "best-{epoch}-{val_loss:.2f}",
        "val_loss",
        mode="min",
    )
    es = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs", name=name),
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=20,
        callbacks=[ckpt, es],
    )

    trainer.fit(mpnn, train_loader, val_loader)
    results = trainer.test(
        mpnn, test_loader, weights_only=False, ckpt_path=ckpt.best_model_path
    )
    print(results)
    print("best checkpoint at:", ckpt.best_model_path)


def main():
    seed_everything(42, workers=True)
    db_root = Path("/home/jackson/rmgpu/rmgdb/db")

    # thermo
    thermo_df = fetch_thermo_training_data(db_root / "thermo.db")
    # TODO: augment with resonance structures to make chemeleon resonance invariant (-ish)
    thermo_upper_bounds = thermo_df[THERMO_TARGETS].max().values
    thermo_lower_bounds = thermo_df[THERMO_TARGETS].min().values
    transform = BoundedUnscaleTransform(
        mean=thermo_df[THERMO_TARGETS].mean().values,
        scale=thermo_df[THERMO_TARGETS].std().values,
        lower_bounds=thermo_lower_bounds,
        upper_bounds=thermo_upper_bounds,
    )
    thermo_mpnn = get_thermo_model(transform)

    # kinetics
    kinetics_df = fetch_kinetics_training_data(db_root / "kinetics.db")
    kinetics_upper_bounds = kinetics_df[KINETICS_TARGETS].max().values
    kinetics_lower_bounds = kinetics_df[KINETICS_TARGETS].min().values
    kinetics_transform = BoundedUnscaleTransform(
        mean=kinetics_df[KINETICS_TARGETS].mean().values,
        scale=kinetics_df[KINETICS_TARGETS].std().values,
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
        "chemeleon_thermo",
        thermo_mpnn,
        thermo_train_loader,
        thermo_val_loader,
        thermo_test_loader,
    )
    fit(
        "chemeleon_kinetics",
        kinetics_mpnn,
        kinetics_train_loader,
        kinetics_val_loader,
        kinetics_test_loader,
    )
