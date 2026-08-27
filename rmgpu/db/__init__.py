"""Database facades for rmgpu.

``Databases`` is the aggregate facade: one sub-facade per rmgdb database
(thermo, kinetics, transport, statmech, solvation) plus the configured
library/family lists that a run wants to use (mirrors the ``database:``
block of the job-03 YAML input schema).

Construction is from a plain config dict::

    Databases.from_config({
        "paths": {"thermo": "...", "kinetics": "...", ...},   # optional
        "thermo_libraries": ["primaryThermoLibrary"],
        "reaction_libraries": ["primaryH2O2"],
        "kinetics_families": ["H_Abstraction", ...],
    })

Sub-facade defaults point at the standard rmgdb build at
/home/jackson/rmgpu/rmgdb/db/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .loaders import ThermoDB, KineticsDB, TransportDB, StatMechDB, SolvationDB

_DEFAULT_PATHS = {
    "thermo": "/home/jackson/rmgpu/rmgdb/db/thermo.db",
    "kinetics": "/home/jackson/rmgpu/rmgdb/db/kinetics.db",
    "transport": "/home/jackson/rmgpu/rmgdb/db/transport.db",
    "statmech": "/home/jackson/rmgpu/rmgdb/db/statmech.db",
    "solvation": "/home/jackson/rmgpu/rmgdb/db/solvation.db",
}


@dataclass
class Databases:
    """Aggregate database facade over the rmgdb SQLite databases."""

    thermo: ThermoDB
    kinetics: KineticsDB
    transport: TransportDB
    statmech: StatMechDB
    solvation: SolvationDB
    # The library/family lists a run requests (mirrors the YAML database: block).
    thermo_libraries: list[str] = field(default_factory=list)
    reaction_libraries: list[str] = field(default_factory=list)
    kinetics_families: list[str] = field(default_factory=list)
    seed_mechanisms: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]] = None) -> "Databases":
        """Construct all facades from a configuration dictionary.

        Keys (all optional):
            paths.thermo / paths.kinetics / paths.transport /
            paths.statmech / paths.solvation - SQLite paths
            thermo_libraries, reaction_libraries, kinetics_families,
            seed_mechanisms - the lists a run wants loaded
        """
        config = config or {}
        paths = {**_DEFAULT_PATHS, **config.get("paths", {})}
        return cls(
            thermo=ThermoDB(db_path=paths["thermo"]),
            kinetics=KineticsDB(db_path=paths["kinetics"]),
            transport=TransportDB(db_path=paths["transport"]),
            statmech=StatMechDB(db_path=paths["statmech"]),
            solvation=SolvationDB(db_path=paths["solvation"]),
            thermo_libraries=list(config.get("thermo_libraries", [])),
            reaction_libraries=list(config.get("reaction_libraries", [])),
            kinetics_families=list(config.get("kinetics_families", [])),
            seed_mechanisms=list(config.get("seed_mechanisms", [])),
        )
