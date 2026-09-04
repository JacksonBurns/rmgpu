"""Command-line interface for rmgpu."""

import os
import sys
from typing import Optional

import click
import yaml

from rmgpu import __version__
from rmgpu.schemas.input import Input
from rmgpu.units import Quantity

NOT_IMPLEMENTED = "not yet implemented"


class QuantityDumper(yaml.SafeDumper):
    pass


def _represent_quantity(dumper: yaml.SafeDumper, q: Quantity):
    return dumper.represent_mapping(
        'tag:yaml.org,2002:map',
        {'value': q._value, 'unit': q._units},
    )


QuantityDumper.add_representer(Quantity, _represent_quantity)


def load_document(path: str) -> dict:
    """Load a YAML file and resolve its extends chain."""
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)
    with open(path) as f:
        doc = yaml.safe_load(f)
    from rmgpu.schemas.input import resolve_extends

    return resolve_extends(doc, base_dir)


def _collect_quantity_errors(doc: dict) -> list[str]:
    """Walk the document and try to parse all quantity-like fields."""
    errors = []
    quantity_fields = [
        'temperature', 'pressure', 'initial_pressure', 'surface_volume_ratio',
        'mbsampling_rate', 'liquid_volume', 'residence_time',
        'inlet_volumetric_flow_rate', 'outlet_volumetric_flow_rate', 'vapor_pressure',
        'staged_temperatures', 'staged_pressures', 'tmin', 'tmax', 'pmin', 'pmax',
        'maximum_grain_size',
    ]

    def _visit(obj, path: str):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key in quantity_fields and value is not None:
                    if isinstance(value, str):
                        try:
                            Quantity.from_string(value)
                        except Exception as e:
                            errors.append(f"Validation error at {current_path}: {e}")
                    elif isinstance(value, dict):
                        _visit(value, current_path)
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            _visit(item, f"{current_path}[{i}]")
                else:
                    _visit(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _visit(item, f"{path}[{i}]")

    _visit(doc, "")
    return errors


@click.group()
def main():
    """rmgpu - Reaction Mechanism Generator (GPU implementation)."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", type=click.Path(dir_okay=True),
              help="Output tree root (default: <input dir>/run_output).")
@click.option("--log-level", type=click.Choice(["DEBUG","INFO","WARNING","ERROR","CRITICAL"], case_sensitive=False),
              default="INFO", help="Logging level.")
@click.option("--log-file", type=click.Path(dir_okay=False), default=None, help="Write logs to file.")
@click.option("--quiet", is_flag=True, help="Suppress INFO output to console (logs go to file if set).")
@click.option("--max-iter", type=int, default=None, help="Maximum CoreEdgeLoop iterations.")
def run(path: str, out: Optional[str], log_level: str, log_file: Optional[str], quiet: bool, max_iter: Optional[int]) -> None:
    """Run the full mechanism generation (job-06 driver): load, build,
    enlarge/simulate/screen to steady state, write the output tree."""
    from rmgpu.logging import setup_logging
    import logging
    level = getattr(logging, log_level.upper(), logging.INFO)
    if quiet:
        level = max(level, logging.WARNING)
    setup_logging(level=level, log_file=log_file)
    try:
        from rmgpu.main import run as run_driver
        from rmgpu.logging import log
        summary = run_driver(path, out_root=out, log_level=level, log_file=log_file, max_iterations=max_iter)
    except Exception as e:
        log.exception("Run failed")
        click.echo(f"Run failed: {e}", err=True)
        sys.exit(1)
    if not quiet:
        click.echo(f"Done. iterations={summary['iterations']} "
                  f"steady_state={summary['steady_state']} "
                  f"core={summary['core_species_count']}spc/"
                  f"{summary['core_reaction_count']}rxn "
                  f"edge={summary['edge_species_count']}spc/"
                  f"{summary['edge_reaction_count']}rxn")
        click.echo(f"Output tree: {summary['out_root']}")


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def validate(path: str) -> None:
    """Validate an input YAML file and report all problems."""
    try:
        doc = load_document(path)
    except Exception as e:
        click.echo(f"Load/extends error: {e}")
        sys.exit(1)

    errors = _collect_quantity_errors(doc)

    try:
        Input(**doc)
    except Exception as e:
        errors.append(str(e))

    if errors:
        for err in errors:
            click.echo(err)
        sys.exit(1)
    else:
        click.echo("Validation passed.")
        sys.exit(0)


@main.command()
@click.option("--out", type=click.Path(dir_okay=False), help="Output file for the JSON schema.")
def schema(out: Optional[str]) -> None:
    """Export the JSON schema for the input document."""
    json_schema = Input.dump_json_schema()
    schema_text = yaml.dump(json_schema, default_flow_style=False)
    if out:
        with open(out, "w") as f:
            f.write(schema_text)
        click.echo(f"Schema written to {out}")
    else:
        click.echo(schema_text)


@main.command()
def version() -> None:
    """Show the rmgpu version."""
    click.echo(__version__)


@main.command(name="import")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--to", required=True, type=click.Path(dir_okay=False), help="Output YAML file path.")
def import_(path: str, to: str) -> None:
    """Import a legacy RMG Python script into YAML."""
    from rmgpu.importer.legacy import import_legacy, LegacyImporterError
    from rmgpu.schemas.input import Input

    try:
        doc = import_legacy(path)
    except LegacyImporterError as e:
        click.echo(f"Import error: {e}", err=True)
        sys.exit(1)

    # Add import notes as YAML comment
    notes = doc.pop('import_notes', [])
    notes_text = ""
    if notes:
        notes_text = "# IMPORT-NOTES:\n"
        for note in notes:
            notes_text += f"# - {note}\n"

    # Validate against schema
    try:
        input_model = Input(**doc)
    except Exception as e:
        click.echo(f"Schema validation error: {e}", err=True)
        sys.exit(1)

    output = input_model.model_dump()

    # Write YAML
    with open(to, 'w') as f:
        if notes_text:
            f.write(notes_text)
        f.write(yaml.dump(output, Dumper=QuantityDumper, default_flow_style=False))

    click.echo(f"Imported {path} -> {to}")


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def export(path: str) -> None:
    """Export a mechanism (not yet implemented)."""
    click.echo(NOT_IMPLEMENTED)


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def diff(path: str) -> None:
    """Diff two mechanisms (not yet implemented)."""
    click.echo(NOT_IMPLEMENTED)


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def inspect(path: str) -> None:
    """Inspect a mechanism (not yet implemented)."""
    click.echo(NOT_IMPLEMENTED)


if __name__ == "__main__":
    main()
