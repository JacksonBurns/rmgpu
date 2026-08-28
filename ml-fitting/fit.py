from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch

from config import KINETICS_TARGETS, THERMO_TARGETS
from data import (
    df_to_chemprop,
    fetch_kinetics_training_data,
    fetch_thermo_training_data,
)
from models import BoundedOutputTransform, get_kinetics_model, get_thermo_model
from viz import plot_multitask_parity


def fit(
    name: str,
    mpnn,
    train_loader,
    val_loader,
    test_loader,
    target_names: tuple[str, ...],
    epochs: int = 100,
):
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
        max_epochs=epochs,
        callbacks=[ckpt, es],
    )

    trainer.fit(mpnn, train_loader, val_loader)

    # Evaluate best checkpoint
    results = trainer.test(
        mpnn, test_loader, weights_only=False, ckpt_path=ckpt.best_model_path
    )
    print(results)
    print("best checkpoint at:", ckpt.best_model_path)

    # Generate predictions on test split
    predictions = trainer.predict(
        mpnn, test_loader, weights_only=False, ckpt_path=ckpt.best_model_path
    )
    y_pred = torch.cat(predictions).detach().cpu().numpy()
    y_true = np.array([d.y for d in test_loader.dataset.data])

    # Plot and save parity figures
    plots_dir = Path(f"plots/{name}")
    plot_multitask_parity(
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        output_dir=plots_dir,
        model_name=name.capitalize(),
    )
    print(f"Parity plots saved to: {plots_dir.resolve()}")


def main():
    pl.seed_everything(42, workers=True)
    db_root = Path("/home/jackson/rmgpu/rmgdb/db")

    # Thermo
    thermo_df = fetch_thermo_training_data(db_root / "thermo.db")
    thermo_df = thermo_df.replace([np.inf, -np.inf], np.nan)
    thermo_upper_bounds = thermo_df[list(THERMO_TARGETS)].max().values
    thermo_lower_bounds = thermo_df[list(THERMO_TARGETS)].min().values

    # TODO: augment training data with resonance structures to try and
    # back in some resonance invariance to the CheMeleon-based thermo
    # model. kinetics models already invariant because of RIGR

    # Kinetics
    kinetics_df = fetch_kinetics_training_data(db_root / "kinetics.db")
    kinetics_df = kinetics_df.replace([np.inf, -np.inf], np.nan)
    kinetics_upper_bounds = kinetics_df[list(KINETICS_TARGETS)].max().values
    kinetics_lower_bounds = kinetics_df[list(KINETICS_TARGETS)].min().values

    (
        thermo_train_loader,
        thermo_val_loader,
        thermo_test_loader,
        thermo_means,
        thermo_stds,
        kinetics_train_loader,
        kinetics_val_loader,
        kinetics_test_loader,
        kinetics_means,
        kinetics_stds,
    ) = df_to_chemprop(thermo_df, kinetics_df)

    thermo_transform = BoundedOutputTransform(
        mean=thermo_means,
        scale=thermo_stds,
        lower_bounds=thermo_lower_bounds,
        upper_bounds=thermo_upper_bounds,
    )
    thermo_mpnn = get_thermo_model(thermo_transform)

    kinetics_transform = BoundedOutputTransform(
        mean=kinetics_means,
        scale=kinetics_stds,
        lower_bounds=kinetics_lower_bounds,
        upper_bounds=kinetics_upper_bounds,
    )
    kinetics_mpnn = get_kinetics_model(kinetics_transform)

    fit(
        "thermo",
        thermo_mpnn,
        thermo_train_loader,
        thermo_val_loader,
        thermo_test_loader,
        THERMO_TARGETS,
    )

    fit(
        "kinetics",
        kinetics_mpnn,
        kinetics_train_loader,
        kinetics_val_loader,
        kinetics_test_loader,
        KINETICS_TARGETS,
        epochs=200,
    )


if __name__ == "__main__":
    main()
