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
def run(path: str) -> None:
    """Load, validate, and print the resolved input document as YAML."""
    try:
        doc = load_document(path)
        input_model = Input(**doc)
    except Exception as e:
        click.echo(f"Validation error: {e}", err=True)
        sys.exit(1)
    output = input_model.model_dump()
    click.echo(yaml.dump(output, Dumper=QuantityDumper, default_flow_style=False))


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
def import_() -> None:
    """Import a legacy RMG Python script into YAML (not yet implemented)."""
    click.echo(NOT_IMPLEMENTED)
    sys.exit(0)


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
