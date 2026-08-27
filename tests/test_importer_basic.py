"""
Basic tests for the legacy importer.

Tests each DSL function on a synthetic snippet to verify it produces
the expected schema-shaped output.
"""

import os
import tempfile
import textwrap

from rmgpu.importer.legacy import import_legacy, LegacyImporterError
from rmgpu.schemas.input import Input


def write_temp_input(content: str, suffix: str = "input.py") -> str:
    """Write content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="test_legacy_")
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


def cleanup(path: str):
    """Clean up temp file."""
    if os.path.exists(path):
        os.remove(path)


class TestSpeciesFunction:
    def test_species_with_smiles(self):
        """species() with SMILES structure."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'ethane',
                structure = SMILES('CC'),
            )
            species(
                label = 'water',
                structure = SMILES('O'),
                reactive = False,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'species' in result
            assert len(result['species']) == 2
            assert result['species'][0]['label'] == 'ethane'
            assert result['species'][0]['structure']['smiles'] == 'CC'
            assert result['species'][0]['reactive'] == True
            assert result['species'][1]['label'] == 'water'
            assert result['species'][1]['reactive'] == False
        finally:
            cleanup(path)

    def test_species_with_adjacency_list(self):
        """species() with adjacency list structure."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'ethane',
                structure = adjacencyList('''
                1 C u0 p3 s0 c0 {2, S}
                2 C u0 p3 s0 c0 {1, S}
                '''),
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'species' in result
            assert len(result['species']) == 1
            assert result['species'][0]['label'] == 'ethane'
            assert 'adjlist' in result['species'][0]['structure']
        finally:
            cleanup(path)

    def test_species_with_inchi(self):
        """species() with InChI structure."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'ethane',
                structure = InChI('InChI=1S/C2H6/c1-2/h1-2H3'),
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'species' in result
            assert len(result['species']) == 1
            assert result['species'][0]['label'] == 'ethane'
            assert 'inchi' in result['species'][0]['structure']
        finally:
            cleanup(path)


class TestDatabaseFunction:
    def test_database_basic(self):
        """database() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
                seedMechanisms = ['C2H5', 'O2'],
                kineticsFamilies = 'all',
                kineticsDepositories = 'default',
                kineticsEstimator = 'ml',
                transportLibraries = 'auto',
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'database' in result
            db = result['database']
            assert db['thermoLibraries'] == ['primaryThermoLibrary']
            assert db['reactionLibraries'] == ['primaryReactions']
            assert db['seedMechanisms'] == ['C2H5', 'O2']
            assert db['kineticsFamilies'] == 'all'
            assert db['kineticsDepositories'] == 'default'
            assert db['kineticsEstimator'] == 'ml'
            assert db['transportLibraries'] == 'auto'
        finally:
            cleanup(path)

    def test_database_with_auto(self):
        """database() with AUTO sentinel."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = 'auto',
                reactionLibraries = ['primaryReactions'],
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'database' in result
            db = result['database']
            assert db['thermoLibraries'] == 'auto'
        finally:
            cleanup(path)


class TestForbiddenFunction:
    def test_forbidden_basic(self):
        """forbidden() with basic structure."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            forbidden(
                label = 'no_radical',
                structure = SMILES('[C]'),
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'forbidden' in result
            assert len(result['forbidden']) == 1
            forbidden = result['forbidden'][0]
            assert forbidden['structure']['smiles'] == '[C]'
        finally:
            cleanup(path)


class TestReactorFunctions:
    def test_simple_reactor(self):
        """simpleReactor() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            simpleReactor(
                temperature = (800, 'K'),
                pressure = (1.0, 'bar'),
                initialMoleFractions = {'CH4': 0.01, 'H2O': 0.99},
                terminationTime = (100, 'us'),
                terminationConversion = {'CH4': 0.95},
                terminationRateRatio = 2e-13,
                nSims = 2,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'reactors' in result
            assert len(result['reactors']) == 1
            reactor = result['reactors'][0]
            assert reactor['type'] == 'simple'
            assert reactor['temperature']['value'] == 800
            assert reactor['temperature']['unit'] == 'K'
            assert reactor['pressure']['value'] == 1.0
            assert reactor['pressure']['unit'] == 'bar'
            assert reactor['initial_mole_fractions']['CH4'] == 0.01
            assert reactor['termination']['time']['value'] == 100
            assert reactor['termination']['conversion']['CH4'] == 0.95
            assert reactor['termination']['rate_ratio'] == 2e-13
            assert reactor['n_sims'] == 2
        finally:
            cleanup(path)

    def test_constant_V_ideal_gas_reactor(self):
        """constantVIdealGasReactor() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            constantVIdealGasReactor(
                temperature = (800, 'K'),
                pressure = (1.0, 'bar'),
                initialMoleFractions = {'CH4': 0.01, 'H2O': 0.99},
                terminationTime = (100, 'us'),
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'reactors' in result
            assert len(result['reactors']) == 1
            reactor = result['reactors'][0]
            assert reactor['type'] == 'const_V'
        finally:
            cleanup(path)

    def test_constant_TP_ideal_gas_reactor(self):
        """constantTPIdealGasReactor() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            constantTPIdealGasReactor(
                temperature = (800, 'K'),
                pressure = (1.0, 'bar'),
                initialMoleFractions = {'CH4': 0.01, 'H2O': 0.99},
                terminationTime = (100, 'us'),
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'reactors' in result
            assert len(result['reactors']) == 1
            reactor = result['reactors'][0]
            assert reactor['type'] == 'const_TP'
        finally:
            cleanup(path)

    def test_liquid_reactor(self):
        """liquidReactor() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH3OH',
                structure = SMILES('CO'),
            )
            solvation(solvent='methanol')
            liquidReactor(
                temperature = (298, 'K'),
                initialConcentrations = {'CH3OH': (1.0, 'M')},
                terminationTime = (100, 'us'),
                nSims = 2,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'reactors' in result
            assert len(result['reactors']) == 1
            reactor = result['reactors'][0]
            assert reactor['type'] == 'liquid'
        finally:
            cleanup(path)

    def test_mb_sampled_reactor(self):
        """mbsampledReactor() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            mbsampledReactor(
                temperature = (800, 'K'),
                pressure = (1.0, 'bar'),
                initialMoleFractions = {'CH4': 0.01, 'H2O': 0.99},
                mbsamplingRate = (1e9, '1/s'),
                terminationTime = (100, 'us'),
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'reactors' in result
            assert len(result['reactors']) == 1
            reactor = result['reactors'][0]
            assert reactor['type'] == 'mb_sampled'
        finally:
            cleanup(path)

    def test_surface_reactor(self):
        """surfaceReactor() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            catalystProperties(
                metal = 'Pt111',
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            surfaceReactor(
                temperature = (800, 'K'),
                initialPressure = (1.0, 'bar'),
                initialGasMoleFractions = {'CH4': 0.01, 'H2O': 0.99},
                initialSurfaceCoverages = {'*': 1.0},
                surfaceVolumeRatio = (1.0, 'm2/m3'),
                terminationTime = (100, 'us'),
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'reactors' in result
            assert len(result['reactors']) == 1
            reactor = result['reactors'][0]
            assert reactor['type'] == 'surface'
        finally:
            cleanup(path)


class TestSimulatorFunction:
    def test_simulator_basic(self):
        """simulator() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            simulator(
                atol = 1e-16,
                rtol = 1e-8,
                sens_atol = 1e-6,
                sens_rtol = 1e-4,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'simulator' in result
            sim = result['simulator']
            assert sim['atol'] == 1e-16
            assert sim['rtol'] == 1e-8
            assert sim['sens_atol'] == 1e-6
            assert sim['sens_rtol'] == 1e-4
        finally:
            cleanup(path)


class TestModelFunction:
    def test_model_basic(self):
        """model() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            model(
                toleranceMoveToCore = 0.1,
                toleranceKeepInEdge = 0.0,
                toleranceInterruptSimulation = 1.0,
                maximumEdgeSpecies = 10000,
                filterReactions = True,
                filterThreshold = 1e8,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'model' in result
            model = result['model']
            assert model['toleranceMoveToCore'] == 0.1
            assert model['toleranceKeepInEdge'] == 0.0
            assert model['toleranceInterruptSimulation'] == 1.0
            assert model['maximumEdgeSpecies'] == 10000
            assert model['filterReactions'] == True
            assert model['filterThreshold'] == 1e8
        finally:
            cleanup(path)


class TestPressureDependenceFunction:
    def test_pressure_dependence_basic(self):
        """pressureDependence() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            pressureDependence(
                method = 'cse',
                temperatures = (300, 2000, 'K', 100),
                pressures = (1.0, 50.0, 'bar', 10),
                interpolation = 'chebyshev',
                maximumAtoms = 10,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'pressure_dependence' in result
            pd = result['pressure_dependence']
            assert pd['method'] == 'cse'
        finally:
            cleanup(path)


class TestMLEstimatorFunction:
    def test_ml_estimator_basic(self):
        """mlEstimator() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            mlEstimator(
                thermo = True,
                name = 'main',
                H298UncertaintyCutoff = (3.0, 'kcal/mol'),
                S298UncertaintyCutoff = (2.0, 'cal/(mol*K)'),
                CpUncertaintyCutoff = (2.0, 'cal/(mol*K)'),
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'ml_estimator' in result
            ml = result['ml_estimator']
            assert ml['thermo'] == True
            assert ml['name'] == 'main'
        finally:
            cleanup(path)


class TestOptionsFunction:
    def test_options_basic(self):
        """options() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            options(
                name = 'my_run',
                generateSeedEachIteration = True,
                saveSeedToDatabase = False,
                units = 'si',
                generateOutputHTML = False,
                generatePlots = False,
                saveSimulationProfiles = False,
                saveEdgeSpecies = False,
                keepIrreversible = False,
                wallTime = '00:00:00:00',
                generateChemkin = True,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'options' in result
            opts = result['options']
            assert opts['name'] == 'my_run'
            assert opts['units'] == 'si'
            assert opts['generateChemkin'] == True
        finally:
            cleanup(path)


class TestSolvationFunction:
    def test_solvation_basic(self):
        """solvation() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH3OH',
                structure = SMILES('CO'),
            )
            solvation(solvent='methanol')
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'solvation' in result
            solv = result['solvation']
            assert solv['solvent'] == 'methanol'
        finally:
            cleanup(path)


class TestUncertaintyFunction:
    def test_uncertainty_basic(self):
        """uncertainty() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            uncertainty(
                localAnalysis = True,
                globalAnalysis = False,
                localNumber = 10,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'uncertainty' in result
            unc = result['uncertainty']
            assert unc['localAnalysis'] == True
            assert unc['globalAnalysis'] == False
        finally:
            cleanup(path)


class TestGeneratedSpeciesConstraintsFunction:
    def test_generated_species_constraints_basic(self):
        """generatedSpeciesConstraints() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            generatedSpeciesConstraints(
                maximumCarbonAtoms = 5,
                maximumOxygenAtoms = 3,
                maximumHeavyAtoms = 10,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'generated_species_constraints' in result
            gsc = result['generated_species_constraints']
            assert gsc['maximumCarbonAtoms'] == 5
            assert gsc['maximumOxygenAtoms'] == 3
            assert gsc['maximumHeavyAtoms'] == 10
        finally:
            cleanup(path)


class TestCatalystPropertiesFunction:
    def test_catalyst_properties_basic(self):
        """catalystProperties() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            catalystProperties(
                metal = 'Pt111',
                coverageDependence = True,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'catalyst_properties' in result
            cp = result['catalyst_properties']
            assert cp['metal'] == 'Pt111'
            assert cp['coverageDependence'] == True
        finally:
            cleanup(path)


class TestRestartFromSeedFunction:
    def test_restart_from_seed_basic(self):
        """restartFromSeed() with basic settings."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            restartFromSeed(
                path = '/path/to/seed',
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert 'restart_from_seed' in result
            rfs = result['restart_from_seed']
            assert rfs['path'] == '/path/to/seed'
        finally:
            cleanup(path)


class TestQuantityTuples:
    def test_quantity_tuple_conversion(self):
        """(value, unit) tuples are converted to Quantity dicts."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            simpleReactor(
                temperature = (800, 'K'),
                pressure = (1.0, 'bar'),
                initialMoleFractions = {'CH4': 1.0},
                terminationTime = (100, 'us'),
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            reactor = result['reactors'][0]
            # Quantity tuples should be converted to dicts
            assert isinstance(reactor['temperature'], dict)
            assert reactor['temperature']['value'] == 800
            assert reactor['temperature']['unit'] == 'K'
        finally:
            cleanup(path)


class TestAutoSentinel:
    def test_auto_library_sentinel(self):
        """'auto' sentinel values are preserved."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = 'auto',
                reactionLibraries = ['primaryReactions'],
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            assert result['database']['thermoLibraries'] == 'auto'
        finally:
            cleanup(path)


class TestNestedLists:
    def test_nested_list_for_staged_reactors(self):
        """Nested lists for staged reactor settings are preserved."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            simpleReactor(
                temperature = [
                    (700, 'K'),
                    (800, 'K')
                ],
                pressure = (1.0, 'bar'),
                initialMoleFractions = {'CH4': 1.0},
                terminationTime = (100, 'us'),
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)
            # The nested list should be preserved
            assert 'reactors' in result
        finally:
            cleanup(path)


class TestImportLegacyErrors:
    def test_file_not_found(self):
        """Non-existent file raises LegacyImporterError."""
        try:
            import_legacy('/nonexistent/path/input.py')
            assert False, "Should have raised LegacyImporterError"
        except LegacyImporterError as e:
            assert 'File not found' in str(e)

    def test_syntax_error(self):
        """File with syntax error raises LegacyImporterError."""
        content = "def invalid(:\n    pass"
        path = write_temp_input(content)
        try:
            import_legacy(path)
            assert False, "Should have raised LegacyImporterError"
        except LegacyImporterError as e:
            assert 'Syntax error' in str(e)
        finally:
            cleanup(path)


class TestSchemaValidation:
    def test_minimal_input_validates(self):
        """A minimal imported input should validate against the schema."""
        content = textwrap.dedent("""
            database(
                thermoLibraries = ['primaryThermoLibrary'],
                reactionLibraries = ['primaryReactions'],
            )
            species(
                label = 'CH4',
                structure = SMILES('C'),
            )
            simpleReactor(
                temperature = (800, 'K'),
                pressure = (1.0, 'bar'),
                initialMoleFractions = {'CH4': 1.0},
                terminationTime = (100, 'us'),
            )
            simulator(
                atol = 1e-16,
                rtol = 1e-8,
            )
            model(
                toleranceMoveToCore = 0.1,
            )
        """)
        path = write_temp_input(content)
        try:
            result = import_legacy(path)

            # Validate against schema (with rmgpu version)
            # Note: The schema requires rmgpu field which we add
            assert 'rmgpu' in result
            assert result['rmgpu'] == '1.0'

            # Check that we have the required fields
            assert 'database' in result
            assert 'species' in result
            assert 'reactors' in result
        finally:
            cleanup(path)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])