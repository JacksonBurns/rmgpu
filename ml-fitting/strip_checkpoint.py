import argparse
from pathlib import Path
import torch

def strip_checkpoint(
    ckpt_path: str | Path,
    output_path: str | Path | None = None,
    keep_lightning_metadata: bool = False,
) -> Path:
    ckpt_path = Path(ckpt_path)
    if output_path is None:
        output_path = ckpt_path.parent / f"{ckpt_path.stem}_deploy.pt"
    else:
        output_path = Path(output_path)

    print(f"Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    initial_size_mb = ckpt_path.stat().st_size / (1024 * 1024)

    # Keys required exclusively for training/resuming
    keys_to_remove = [
        "optimizer_states",
        "lr_schedulers",
        "callbacks",
        "loops",
        "native_amp_scaling_state",
        "hparams_name",
    ]

    if not keep_lightning_metadata:
        keys_to_remove.extend(["epoch", "global_step"])

    for key in keys_to_remove:
        checkpoint.pop(key, None)

    # Save lightweight dictionary containing only state_dict and hyperparameters
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    final_size_mb = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - (final_size_mb / initial_size_mb)) * 100

    print(f"Saved:   {output_path}")
    print(f"Size:    {initial_size_mb:.2f} MB -> {final_size_mb:.2f} MB ({reduction:.1f}% reduction)\n")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Strip training state from PyTorch Lightning checkpoints.")
    parser.add_argument(
        "checkpoints",
        nargs="*",
        help="Path(s) to checkpoint .ckpt files. If empty, scans the checkpoints/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deploy_models",
        help="Directory to save stripped model files (default: deploy_models/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.checkpoints:
        ckpt_files = [Path(p) for p in args.checkpoints]
    else:
        ckpt_files = list(Path("checkpoints").glob("**/*.ckpt"))

    if not ckpt_files:
        print("No .ckpt files found.")
        return

    for ckpt in ckpt_files:
        out_file = out_dir / f"{ckpt.parent.name}_{ckpt.stem}_deploy.pt"
        strip_checkpoint(ckpt, out_file)

if __name__ == "__main__":
    main()
