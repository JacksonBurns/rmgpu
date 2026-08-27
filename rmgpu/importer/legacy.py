"""
Legacy RMG Python input importer.

Converts legacy RMG .py input files into schema-shaped dictionaries
for rmgpu's YAML-based input system. Uses AST parsing only (no exec).
"""

import ast
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rmgpu.schemas.input import Input


# DSL function mapping: DSL name -> (schema_key, schema_type)
# DSL functions that don't map to a schema key are marked with None
DSL_MAPPINGS = {
    # Database functions
    'database': ('database', 'dict'),
    'thermoLibraries': ('thermo_libraries', 'list'),
    'reactionLibraries': ('reaction_libraries', 'list'),
    'seedMechanisms': ('seed_mechanisms', 'list'),
    'kineticsFamilies': ('kinetics_families', 'string_or_list'),
    'kineticsDepositories': ('kinetics_depositories', 'string_or_list'),
    'kineticsEstimator': ('kinetics_estimator', 'string'),
    'transportLibraries': ('transport_libraries', 'list'),

    # Species functions
    'species': ('species', 'list'),
    'forbidden': ('forbidden', 'list'),
    'SMILES': ('structure_smiles', 'string'),
    'InChI': ('structure_inchi', 'string'),
    'adjacencyList': ('structure_adjlist', 'string'),
    'adjacencyListGroup': ('structure_group_adjlist', 'string'),

    # Reactor functions
    'simpleReactor': ('reactors', 'list'),
    'constantVIdealGasReactor': ('reactors', 'list'),
    'constantTPIdealGasReactor': ('reactors', 'list'),
    'liquidReactor': ('reactors', 'list'),
    'mbsampledReactor': ('reactors', 'list'),
    'surfaceReactor': ('reactors', 'list'),

    # Simulator/model functions
    'simulator': ('simulator', 'dict'),
    'model': ('model', 'dict'),
    'pressureDependence': ('pressure_dependence', 'dict'),
    'mlEstimator': ('ml_estimator', 'dict'),
    'options': ('options', 'dict'),
    'solvation': ('solvation', 'dict'),
    'uncertainty': ('uncertainty', 'dict'),
    'generatedSpeciesConstraints': ('generated_species_constraints', 'dict'),
    'quantumMechanics': ('quantum_mechanics', 'dict'),
    'restartFromSeed': ('restart_from_seed', 'dict'),
    'catalystProperties': ('catalyst_properties', 'dict'),

    # Quantity/unit functions (handled inline)
    'Quantity': None,
    'Energy': None,
    'RateCoefficient': None,
    'Concentration': None,

    # Structure helpers
    'fragment_adj': None,
    'fragment_SMILES': None,
    'fragment_adjacencyList': None,
    'coreSpeciesFile': None,
    'react': None,

    # Liquid/surface specific
    'liquidVolumetricMassTransferCoefficientPowerLaw': None,
    'constantTVLiquidReactor': ('reactors', 'list'),
    'liquidSurfaceReactor': ('reactors', 'list'),
}

# Sentinel values that have special meaning
AUTO_TOKENS = {'auto', 'AUTO'}
PAH_LIB_TOKENS = {'<PAH_libs>', 'PAH_LIBS'}


class LegacyImporterError(Exception):
    """Raised when the legacy importer cannot process a file."""
    pass


class ImportNote:
    """A note about something the importer couldn't express."""

    def __init__(self, function_name: str, line_number: int, message: str):
        self.function_name = function_name
        self.line_number = line_number
        self.message = message

    def __str__(self):
        return f"Line {self.line_number}: {self.function_name} - {self.message}"


def _node_to_literal(node: ast.AST) -> Any:
    """Convert an AST node to a literal value if possible."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _convert_quantity(value: Any) -> Optional[Dict[str, Any]]:
    """Convert a (value, unit) tuple to a Quantity dict."""
    if isinstance(value, tuple) and len(value) == 2:
        val, unit = value
        if isinstance(val, (int, float)) and isinstance(unit, str):
            return {"value": val, "unit": unit}
    return None


def _parse_structure(structure_node: ast.AST) -> Optional[Dict[str, str]]:
    """Parse a structure specification node (SMILES, InChI, or adjacency list)."""
    if not isinstance(structure_node, ast.Call):
        return None

    func_name = getattr(structure_node.func, 'id', None)
    if func_name is None:
        # Handle dotted names like Molecule.from_smiles
        if isinstance(structure_node.func, ast.Attribute):
            func_name = structure_node.func.attr

    if func_name is None:
        return None

    if not structure_node.args:
        return None

    str_node = structure_node.args[0]
    str_val = _node_to_literal(str_node)
    if not isinstance(str_val, str):
        return None

    # Map function names to structure types
    if func_name in ('SMILES', 'smiles', 'from_smiles'):
        return {"smiles": str_val}
    elif func_name in ('InChI', 'inchi', 'from_inchi'):
        return {"inchi": str_val}
    elif func_name in ('adjacencyList', 'adjacency_list', 'from_adjacency_list'):
        return {"adjlist": str_val}
    elif func_name in ('adjacencyListGroup', 'adjacency_list_group', 'from_adjacency_list'):
        return {"group_adjlist": str_val}
    elif func_name in ('SMARTS', 'smarts'):
        return {"smarts": str_val}

    return None


def _parse_termination_conditions(kwargs: Dict[str, ast.AST]) -> Optional[Dict[str, Any]]:
    """Parse termination condition kwargs into a TerminationCondition dict."""
    result = {}

    if 'terminationConversion' in kwargs:
        conv_node = kwargs['terminationConversion']
        conv_val = _node_to_literal(conv_node)
        if isinstance(conv_val, dict):
            result['conversion'] = conv_val

    if 'terminationTime' in kwargs:
        time_node = kwargs['terminationTime']
        time_val = _node_to_literal(time_node)
        qty = _convert_quantity(time_val)
        if qty:
            result['time'] = qty

    if 'terminationRateRatio' in kwargs:
        ratio_node = kwargs['terminationRateRatio']
        ratio_val = _node_to_literal(ratio_node)
        if isinstance(ratio_val, (int, float)):
            result['rate_ratio'] = ratio_val

    if not result:
        return None

    return result


class LegacyVisitor(ast.NodeVisitor):
    """AST visitor that parses legacy RMG input files into schema-shaped dicts."""

    def __init__(self):
        self.result = {}
        self.import_notes = []
        self.species_list = []
        self.forbidden_list = []
        self.reactors_list = []
        self.current_context = None  # 'species', 'forbidden', 'reactor', etc.
        self._skip_nested_structures = False

    def visit_Call(self, node: ast.Call):
        """Visit a function call."""
        func_name = self._get_func_name(node)
        if func_name is None:
            self._generic_visit(node)
            return

        mapping = DSL_MAPPINGS.get(func_name)
        if mapping is None:
            # Check if it's a structure function
            if func_name in ('SMILES', 'InChI', 'adjacencyList', 'adjacencyListGroup', 'SMARTS'):
                if not self._skip_nested_structures:
                    self._visit_structure_call(node, func_name)
            else:
                # Unhandled function - add import note
                self._add_import_note(func_name, node.lineno, f"Unhandled function: {func_name}")
            return

        schema_key, schema_type = mapping

        if schema_type == 'list':
            self._visit_list_function(node, func_name, schema_key)
        elif schema_type == 'dict':
            self._visit_dict_function(node, func_name, schema_key)
        elif schema_type == 'string':
            self._visit_string_function(node, func_name, schema_key)
        elif schema_type == 'string_or_list':
            self._visit_string_or_list_function(node, func_name, schema_key)
        elif schema_type == 'structure_smiles':
            self._visit_structure_call(node, func_name)

        # Don't visit nested calls for species/forbidden/reactor functions
        # because they handle their nested structure calls internally
        if func_name not in ('SMILES', 'InChI', 'adjacencyList', 'adjacencyListGroup', 'SMARTS',
                             'species', 'forbidden',
                             'simpleReactor', 'constantVIdealGasReactor', 'constantTPIdealGasReactor',
                             'liquidReactor', 'mbsampledReactor', 'surfaceReactor',
                             'constantTVLiquidReactor', 'liquidSurfaceReactor'):
            self._generic_visit(node)

    def _get_func_name(self, node: ast.Call) -> Optional[str]:
        """Extract the function name from a call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _add_import_note(self, function_name: str, line_number: int, message: str):
        """Add an import note for something we couldn't express."""
        self.import_notes.append(ImportNote(function_name, line_number, message))

    def _visit_list_function(self, node: ast.Call, func_name: str, schema_key: str):
        """Visit a function that adds to a list (species, forbidden, reactors)."""
        if func_name == 'species':
            self._visit_species(node)
        elif func_name == 'forbidden':
            self._visit_forbidden(node)
        elif func_name in ('simpleReactor', 'constantVIdealGasReactor', 'constantTPIdealGasReactor',
                           'liquidReactor', 'mbsampledReactor', 'surfaceReactor',
                           'constantTVLiquidReactor', 'liquidSurfaceReactor'):
            self._visit_reactor(node, func_name)
        else:
            # Generic list function - just add the call info
            if schema_key not in self.result:
                self.result[schema_key] = []
            # Try to parse arguments
            args = []
            for arg in node.args:
                lit = _node_to_literal(arg)
                args.append(lit)
            self.result[schema_key].append({"name": func_name, "args": args})

    def _visit_dict_function(self, node: ast.Call, func_name: str, schema_key: str):
        """Visit a function that sets a dict (database, simulator, model, etc.)."""
        if schema_key not in self.result:
            self.result[schema_key] = {}

        # Parse keyword arguments
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue  # Skip **kwargs
            val_node = kw.value
            lit = _node_to_literal(val_node)
            if lit is not None:
                kwargs[kw.arg] = lit

        # Merge into result
        self.result[schema_key].update(kwargs)

    def _visit_string_function(self, node: ast.Call, func_name: str, schema_key: str):
        """Visit a function that sets a string value."""
        if not node.args:
            return
        lit = _node_to_literal(node.args[0])
        if lit is not None:
            self.result[schema_key] = lit

    def _visit_string_or_list_function(self, node: ast.Call, func_name: str, schema_key: str):
        """Visit a function that sets a string or list value."""
        if not node.args:
            return
        lit = _node_to_literal(node.args[0])
        if lit is not None:
            self.result[schema_key] = lit

    def _visit_species(self, node: ast.Call):
        """Visit a species() call."""
        # Handle both positional and keyword arguments
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = kw.value

        # Get label from positional arg or keyword arg
        if len(node.args) >= 1:
            label_node = node.args[0]
            label = _node_to_literal(label_node)
        elif 'label' in kwargs:
            label = _node_to_literal(kwargs['label'])
        else:
            return

        if not isinstance(label, str):
            return

        # Get structure from positional arg or keyword arg
        if len(node.args) >= 2:
            structure_node = node.args[1]
        elif 'structure' in kwargs:
            structure_node = kwargs['structure']
        else:
            return

        structure = _parse_structure(structure_node)
        if structure is None:
            # Try to parse from a string arg
            structure_val = _node_to_literal(structure_node)
            if isinstance(structure_val, str):
                structure = {"smiles": structure_val}
            else:
                return

        # Parse other kwargs for reactive, etc.
        other_kwargs = {k: v for k, v in kwargs.items() if k not in ('label', 'structure')}
        parsed_kwargs = {}
        for kw_name, kw_node in other_kwargs.items():
            lit = _node_to_literal(kw_node)
            if lit is not None:
                parsed_kwargs[kw_name] = lit

        species_entry = {
            "label": label,
            "structure": structure,
            "reactive": parsed_kwargs.get('reactive', True),
        }
        if 'thermo' in parsed_kwargs:
            species_entry['thermo'] = parsed_kwargs['thermo']
        if 'kinetics' in parsed_kwargs:
            species_entry['kinetics'] = parsed_kwargs['kinetics']
        if 'constraints' in parsed_kwargs:
            species_entry['constraints'] = parsed_kwargs['constraints']

        self.species_list.append(species_entry)

    def _visit_forbidden(self, node: ast.Call):
        """Visit a forbidden() call."""
        # Handle both positional and keyword arguments
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = kw.value

        # Get label from positional arg or keyword arg
        if len(node.args) >= 1:
            label_node = node.args[0]
            label = _node_to_literal(label_node)
        elif 'label' in kwargs:
            label = _node_to_literal(kwargs['label'])
        else:
            return

        if not isinstance(label, str):
            return

        # Get structure from positional arg or keyword arg
        if len(node.args) >= 2:
            structure_node = node.args[1]
        elif 'structure' in kwargs:
            structure_node = kwargs['structure']
        else:
            return

        structure = _parse_structure(structure_node)
        if structure is None:
            structure_val = _node_to_literal(structure_node)
            if isinstance(structure_val, str):
                structure = {"smiles": structure_val}
            else:
                return

        forbidden_entry = {
            "structure": structure,
            "reason": None,
        }
        self.forbidden_list.append(forbidden_entry)

    def _visit_reactor(self, node: ast.Call, func_name: str):
        """Visit a reactor function call."""
        # Determine reactor type from function name
        reactor_type_map = {
            'simpleReactor': 'simple',
            'constantVIdealGasReactor': 'const_V',
            'constantTPIdealGasReactor': 'const_TP',
            'liquidReactor': 'liquid',
            'mbsampledReactor': 'mb_sampled',
            'surfaceReactor': 'surface',
            'constantTVLiquidReactor': 'const_V_liquid',
            'liquidSurfaceReactor': 'liquid_surface',
        }
        reactor_type = reactor_type_map.get(func_name, 'unknown')

        reactor = {"type": reactor_type}

        # Parse keyword arguments first (more common in legacy files)
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = kw.value

        # Parse temperature
        if 'temperature' in kwargs:
            temp_node = kwargs['temperature']
            temp_val = _node_to_literal(temp_node)
            qty = _convert_quantity(temp_val)
            if qty:
                reactor['temperature'] = qty
        elif len(node.args) >= 1:
            temp_val = _node_to_literal(node.args[0])
            qty = _convert_quantity(temp_val)
            if qty:
                reactor['temperature'] = qty

        # Parse pressure (for gas reactors)
        if reactor_type in ('simple', 'const_V', 'const_TP', 'mb_sampled'):
            if 'pressure' in kwargs:
                press_node = kwargs['pressure']
                press_val = _node_to_literal(press_node)
                qty = _convert_quantity(press_val)
                if qty:
                    reactor['pressure'] = qty
            elif len(node.args) >= 2:
                press_val = _node_to_literal(node.args[1])
                qty = _convert_quantity(press_val)
                if qty:
                    reactor['pressure'] = qty

        # Parse initialMoleFractions or initialConcentrations
        if 'initialMoleFractions' in kwargs:
            imf_node = kwargs['initialMoleFractions']
            imf_val = _node_to_literal(imf_node)
            if isinstance(imf_val, dict):
                reactor['initial_mole_fractions'] = imf_val
        elif 'initialConcentrations' in kwargs:
            ic_node = kwargs['initialConcentrations']
            ic_val = _node_to_literal(ic_node)
            if isinstance(ic_val, dict):
                reactor['initial_concentrations'] = ic_val
        elif len(node.args) >= 3:
            imf_node = node.args[2]
            imf_val = _node_to_literal(imf_node)
            if isinstance(imf_val, dict):
                reactor['initial_mole_fractions'] = imf_val

        # Parse other kwargs
        termination = {}
        for kw_name, val_node in kwargs.items():
            if kw_name in ('temperature', 'pressure', 'initialMoleFractions', 'initialConcentrations'):
                continue
            lit = _node_to_literal(val_node)

            if kw_name == 'terminationConversion':
                if isinstance(lit, dict):
                    termination['conversion'] = lit
            elif kw_name == 'terminationTime':
                qty = _convert_quantity(lit)
                if qty:
                    termination['time'] = qty
            elif kw_name == 'terminationRateRatio':
                if isinstance(lit, (int, float)):
                    termination['rate_ratio'] = lit
            elif kw_name == 'nSims':
                if isinstance(lit, int):
                    reactor['n_sims'] = lit
            elif kw_name == 'sensitivity':
                reactor['sensitivity'] = lit
            elif kw_name == 'sensitivityThreshold':
                reactor['sensitivity_threshold'] = lit
            elif kw_name == 'constantSpecies':
                reactor['constant_species'] = lit
            elif kw_name == 'balanceSpecies':
                reactor['balance_species'] = lit
            elif kw_name == 'mbsamplingRate':
                qty = _convert_quantity(lit)
                if qty:
                    reactor['mbsampling_rate'] = qty
            elif kw_name == 'initialPressure':
                qty = _convert_quantity(lit)
                if qty:
                    reactor['initial_pressure'] = qty
            elif kw_name == 'initialGasMoleFractions':
                reactor['initial_gas_mole_fractions'] = lit
            elif kw_name == 'initialSurfaceCoverages':
                reactor['initial_surface_coverages'] = lit
            elif kw_name == 'surfaceVolumeRatio':
                qty = _convert_quantity(lit)
                if qty:
                    reactor['surface_volume_ratio'] = qty
            elif kw_name == 'liquidVolume':
                qty = _convert_quantity(lit)
                if qty:
                    reactor['liquid_volume'] = qty
            elif kw_name == 'residenceTime':
                qty = _convert_quantity(lit)
                if qty:
                    reactor['residence_time'] = qty

        if termination:
            reactor['termination'] = termination

        self.reactors_list.append(reactor)

    def _visit_structure_call(self, node: ast.Call, func_name: str):
        """Visit a structure function call (SMILES, InChI, adjacencyList)."""
        pass  # Structure parsing is handled in _parse_structure

    def _generic_visit(self, node: ast.AST):
        """Fallback visit that processes children, skipping nested calls."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                continue  # Don't visit nested calls
            self.visit(child)

    def get_result(self) -> Tuple[Dict[str, Any], List[ImportNote]]:
        """Return the parsed result and import notes."""
        # Add species, forbidden, reactors to result
        if self.species_list:
            self.result['species'] = self.species_list
        if self.forbidden_list:
            self.result['forbidden'] = self.forbidden_list
        if self.reactors_list:
            self.result['reactors'] = self.reactors_list

        return self.result, self.import_notes


def _parse_quantity_arg(node: ast.AST) -> Optional[Dict[str, Any]]:
    """Parse a Quantity() call node into a quantity dict."""
    if not isinstance(node, ast.Call):
        return None
    if len(node.args) < 2:
        return None
    val_node = node.args[0]
    unit_node = node.args[1]
    val = _node_to_literal(val_node)
    unit = _node_to_literal(unit_node)
    if isinstance(val, (int, float)) and isinstance(unit, str):
        return {"value": val, "unit": unit}
    return None


def import_legacy(path: str) -> Dict[str, Any]:
    """
    Import a legacy RMG Python input file into a schema-shaped dictionary.

    Args:
        path: Path to the legacy .py input file.

    Returns:
        A dictionary with schema-shaped keys and values, plus an 'import_notes'
        key containing any notes about things that couldn't be expressed.

    Raises:
        LegacyImporterError: If the file cannot be parsed or imported.
    """
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise LegacyImporterError(f"File not found: {path}")

    try:
        with open(path, 'r') as f:
            source = f.read()
    except IOError as e:
        raise LegacyImporterError(f"Could not read file: {e}")

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        raise LegacyImporterError(f"Syntax error in {path}: {e}")

    # Walk the AST to collect function calls
    visitor = LegacyVisitor()
    visitor.visit(tree)

    result, import_notes = visitor.get_result()

    # Add version (required by schema)
    result['rmgpu'] = '1.0'

    # Add import notes
    if import_notes:
        result['import_notes'] = [str(note) for note in import_notes]

    return result


def _extract_function_calls(tree: ast.AST) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extract function calls from the AST with their arguments.

    Returns:
        List of (function_name, args_dict) tuples.
    """
    calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            continue

        # Collect positional args
        args = []
        for arg in node.args:
            lit = _node_to_literal(arg)
            if lit is not None:
                args.append(lit)

        # Collect keyword args
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            lit = _node_to_literal(kw.value)
            if lit is not None:
                kwargs[kw.arg] = lit

        calls.append((func_name, {"args": args, "kwargs": kwargs}))

    return calls


def dump_legacy(path: str) -> Dict[str, Any]:
    """
    Dump a legacy RMG Python input file as raw parsed structure (JSON-able).

    This is the ground truth for the gate's diff - it captures everything
    the AST visitor sees without any schema transformation.

    Args:
        path: Path to the legacy .py input file.

    Returns:
        A JSON-serializable dictionary with all parsed function calls.
    """
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise LegacyImporterError(f"File not found: {path}")

    try:
        with open(path, 'r') as f:
            source = f.read()
    except IOError as e:
        raise LegacyImporterError(f"Could not read file: {e}")

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        raise LegacyImporterError(f"Syntax error in {path}: {e}")

    calls = _extract_function_calls(tree)

    return {
        "file": os.path.basename(path),
        "functions": [
            {"name": name, "args": args_data["args"], "kwargs": args_data["kwargs"]}
            for name, args_data in calls
        ]
    }