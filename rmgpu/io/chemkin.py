"""Chemkin file writer for rmgpu mechanisms.

Produces chem.inp, chem_annotated.inp, and species_dictionary.txt in a
format compatible with RMG-Py's writer.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from rmgpu.schemas.mechanism import MechanismArtifact


def _format_fortran(value: float, precision: int = 3) -> str:
    """Format a float in Fortran-style scientific notation (e.g., 1.00D+03)."""
    if value == 0:
        return f"0.00{'D'+str(0).zfill(2)}"
    sign = "-" if value < 0 else ""
    value = abs(value)
    exponent = int(round(__import__("math").log10(value)))
    mantissa = value / (10 ** exponent)
    mantissa_str = f"{mantissa:.{precision}f}"
    if len(mantissa_str) > precision + 2:  # x.yyy
        mantissa_str = mantissa_str[:precision + 2]
    return f"{sign}{mantissa_str}{'D'+str(exponent).zfill(2).replace('10', '')}"


def _parse_formula(formula: str) -> Dict[str, int]:
    """Parse a chemical formula string into a dict of element counts."""
    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    elements = {}
    for match in pattern.finditer(formula):
        elem = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        elements[elem] = elements.get(elem, 0) + count
    return elements


def _write_thermo_species_line(species: dict, verbose: bool) -> str:
    """Write a single thermo species line in Chemkin format.
    
    Format: SPECIES_NAME  FORMULA  ELEMENTS...  [G]  Tmin  Tmax  Tint
            a0_high a1_high a2_high a3_high a4_high
            a5_high a6_high a0_low a1_low a2_low
            a3_low a4_low a5_low a6_low
    """
    name = species["label"][:16]
    comment = species.get("comment", "")[:20]
    formula = species.get("formula", "")
    elements = species.get("elements", {})
    Tmin = species.get("Tmin", 300.0)
    Tmax = species.get("Tmax", 5000.0)
    Tint = species.get("Tint", 1000.0)
    
    # Elements section: C H N O etc. with counts
    elem_str = ""
    for elem, count in elements.items():
        elem_str += f"{elem}{count}"
    elem_str = elem_str.ljust(30)[:30]  # Chemkin limit
    
    # Format: name(16) + comment(6) + elements(30) + G(1) + Tmin(10) + Tmax(10) + Tint(10)
    # = 16 + 6 + 30 + 1 + 10 + 10 + 10 = 83 chars
    name_part = name.ljust(16)[:16]
    comment_part = comment.ljust(6)[:6]
    elem_part = elem_str
    phase_part = "G"
    tmin_part = f"{int(Tmin):>10d}"
    tmax_part = f"{int(Tmax):>10d}"
    tint_part = f"{int(Tint):>10d}"
    
    line1 = f"{name_part}{comment_part}{elem_part}{phase_part}{tmin_part}{tmax_part}{tint_part}"
    
    coeffs = species.get("coeffs", [[0.0]*7 for _ in range(2)])
    a_high = coeffs[1]  # high-T first
    a_low = coeffs[0]   # low-T second
    
    line2 = f"{a_high[0]:>15.5e}{a_high[1]:>15.5e}{a_high[2]:>15.5e}{a_high[3]:>15.5e}{a_high[4]:>15.5e}"
    line3 = f"{a_high[5]:>15.5e}{a_high[6]:>15.5e}{a_low[0]:>15.5e}{a_low[1]:>15.5e}{a_low[2]:>15.5e}"
    line4 = f"{a_low[3]:>15.5e}{a_low[4]:>15.5e}{a_low[5]:>15.5e}{a_low[6]:>15.5e}"
    
    return f"{line1}\n{line2}\n{line3}\n{line4}\n"


def _write_reaction_line(reaction: dict, verbose: bool) -> str:
    """Write a reaction line in Chemkin format.
    
    For Arrhenius: REACTANTS=PRODUCTS  A  n  Ea
    For falloff: adds LOW and EFF lines.
    """
    reactants = reaction["reactants"]
    products = reaction["products"]
    rxn_str = "".join(reactants) + "=" + "".join(products)
    
    rate = reaction["rate"]
    rate_type = rate.get("type", "arrhenius")
    params = rate.get("params", {})
    
    A = params.get("A", 0.0)
    n = params.get("n", 0.0)
    Ea = params.get("Ea", 0.0) / 4184.0  # Convert J/mol to kcal/mol for Chemkin
    
    if verbose:
        prefix = "! "
    else:
        prefix = ""
    
    line = f"{prefix}{rxn_str:<60}{A:>10.3e} {n:>10.3f} {Ea:>10.3f}\n"
    
    if rate_type == "falloff":
        A_low = params.get("A_low", A)
        n_low = params.get("n_low", n)
        Ea_low = params.get("Ea_low", Ea * 1000) / 1000.0
        
        line += f"{prefix}    LOW/ {A_low:>10.3e} {n_low:>10.3f} {Ea_low:>10.3f}/\n"
        
        efficiencies = params.get("efficiencies", {})
        if efficiencies:
            eff_str = "    EFF/"
            for label, eff in efficiencies.items():
                eff_str += f" {label} {eff} /"
            line += eff_str + "\n"
    
    return line


def _write_species_dictionary(species: List[dict], path: str) -> None:
    """Write the species dictionary file (adjacency lists)."""
    with open(path, "w") as f:
        for species in species:
            label = species["label"]
            smiles = species.get("smiles", "")
            adjlist = species.get("adjlist", "")
            f.write(f"{label}\n")
            f.write(f"    SMILES: {smiles}\n")
            f.write(f"{adjlist}\n\n")


def write_chemkin(mechanism: MechanismArtifact, path: str) -> None:
    """Write Chemkin files from a mechanism artifact.
    
    Creates three files in the given directory:
    - chem.inp: plain Chemkin format
    - chem_annotated.inp: Chemkin format with comments
    - species_dictionary.txt: species with adjacency lists
    """
    os.makedirs(path, exist_ok=True)
    
    core = mechanism.core
    species_list: List[dict] = []
    elements_in_use = set()
    
    # Process species
    for sp in core.species:
        formula = sp.formula or ""
        elements = _parse_formula(formula)
        elements_in_use.update(elements.keys())
        
        # Get thermo coefficients (simplified - will need real NASA7 coeffs)
        thermo = sp.thermo
        coeffs = [[0.0] * 7 for _ in range(2)]
        if thermo.model == "nasa7" and thermo.Cp:
            # Simplified: use Cp values if available
            if len(thermo.Cp) >= 7:
                coeffs[0] = thermo.Cp[:7]
                coeffs[1] = thermo.Cp[:7]
        
        species_data: dict = {
            "label": sp.label,
            "formula": formula,
            "elements": elements,
            "coeffs": coeffs,
            "smiles": sp.smiles or "",
            "adjlist": sp.adjlist or "",
            "Tmin": 300.0,
            "Tmax": 5000.0,
            "Tint": 1000.0,
            "comment": "",
        }
        species_list.append(species_data)
    
    # Process reactions
    reaction_list: List[dict] = []
    for rxn in core.reactions:
        reactants_str = []
        products_str = []
        for r in rxn.reactants:
            reactants_str.append(r)
        for p in rxn.products:
            products_str.append(p)
        
        rate_type = rxn.rate.type
        params = rxn.rate.params or {}
        
        reaction_data: dict = {
            "label": rxn.label,
            "reactants": reactants_str,
            "products": products_str,
            "rate": {
                "type": rate_type,
                "params": params,
            },
            "family": rxn.family or "",
        }
        reaction_list.append(reaction_data)
    
    # Write chem.inp
    chem_path = os.path.join(path, "chem.inp")
    with open(chem_path, "w") as f:
        # Elements section
        elements_sorted = sorted(elements_in_use)
        f.write("ELEMENTS\n")
        for elem in elements_sorted:
            f.write(f"\t{elem}\n")
        f.write("END\n\n")
        
        # Species section
        f.write("SPECIES\n")
        for sp in species_list:
            f.write(_write_thermo_species_line(sp, verbose=False))
        f.write("END\n\n")
        
        # Reactions section
        f.write("REACTIONS\n")
        for rxn in reaction_list:
            f.write(_write_reaction_line(rxn, verbose=False))
        f.write("END\n\n")
        
        # Termination
        f.write("TERMINATION\n")
        f.write("    H,H2,H2O\n")
        f.write("END\n\n")
        
        # Initial states
        f.write("INITIAL\n")
        f.write("    1 0.0 1000. 101325.\n")
        f.write("    2 0.0 1000. 101325.\n")
        f.write("END\n\n")
        
        # Initial species
        f.write("INITIAL-SPECIES\n")
        f.write("END\n\n")
        
        # End
        f.write("END\n")
    
    # Write chem_annotated.inp (same but with comments)
    annotated_path = os.path.join(path, "chem_annotated.inp")
    with open(annotated_path, "w") as f:
        elements_sorted = sorted(elements_in_use)
        f.write("! ELEMENTS\n")
        for elem in elements_sorted:
            f.write(f"\t{elem}\n")
        f.write("! END\n\n")
        
        f.write("! SPECIES\n")
        for sp in species_list:
            f.write(_write_thermo_species_line(sp, verbose=True))
        f.write("! END\n\n")
        
        f.write("! REACTIONS\n")
        for rxn in reaction_list:
            f.write(_write_reaction_line(rxn, verbose=True))
        f.write("! END\n\n")
        
        f.write("! TERMINATION\n")
        f.write("!     H,H2,H2O\n")
        f.write("! END\n\n")
        
        f.write("! INITIAL\n")
        f.write("!     1 0.0 1000. 101325.\n")
        f.write("!     2 0.0 1000. 101325.\n")
        f.write("! END\n\n")
        
        f.write("! INITIAL-SPECIES\n")
        f.write("! END\n\n")
        
        f.write("! END\n")
    
    # Write species_dictionary.txt
    dict_path = os.path.join(path, "species_dictionary.txt")
    _write_species_dictionary(species_list, dict_path)
