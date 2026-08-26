"""Kinetics retrieval logic for rmgpu.

Provides a thin interface to look up reaction kinetics from libraries first,
falling back to the ML estimator (job 04) if not found in any library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rmgpu.db.loaders import KineticsDB


@dataclass
class KineticsLookupResult:
    """Result of a kinetics lookup."""
    found: bool
    source: str = ""  # "library" or "ml"
    reaction: Optional[dict] = None


def lookup_kinetics(reaction: dict, kinetics_db: KineticsDB) -> KineticsLookupResult:
    """
    Look up kinetics for a reaction.
    
    Strategy:
    1. Check if reaction exists in kinetics libraries (substructure match)
    2. If not found, use ML estimator (TODO: job-04)
    
    Args:
        reaction: dict with 'reactants' and 'products' keys (list of labels)
        kinetics_db: KineticsDB facade instance
        
    Returns:
        KineticsLookupResult with found status and source
    """
    # 1. Try to find in libraries
    label = _make_label(reaction)
    result = kinetics_db.get_reaction_by_label(label)
    
    if result is not None:
        return KineticsLookupResult(found=True, source="library", reaction=result)
    
    # 2. TODO(job-04): Use ML estimator if not in libraries
    # raise NotImplementedError("ML kinetics lookup not implemented yet - see job-04")
    
    # For now, report not found
    return KineticsLookupResult(found=False, source="", reaction=None)


def _make_label(reaction: dict) -> str:
    """
    Create a label from a reaction dict.
    
    Format: reactant1+reactant2=product1+product2
    """
    reactants = reaction.get("reactants", [])
    products = reaction.get("products", [])
    reactant_str = "+".join(reactants)
    product_str = "+".join(products)
    return f"{reactant_str}={product_str}"
