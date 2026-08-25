import click

@click.group()
def main():
    """rmgpu - Reaction Mechanism Generator (GPU implementation)."""
    pass

@main.command()
def run():
    """Run mechanism generation."""
    click.echo("not yet implemented")

@main.command()
def validate():
    """Validate input files."""
    click.echo("not yet implemented")

@main.command()
def version():
    """Show version."""
    from rmgpu import __version__
    click.echo(__version__)

if __name__ == '__main__':
    main()
