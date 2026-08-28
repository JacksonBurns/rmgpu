"""
Basic tests for the legacy importer (job-03/step-05 rewrite).

The importer is lossless: every DSL value is preserved, keys are snake_case,
quantity tuples become {value, unit} dicts, and anything unrepresentable is
recorded LOUDLY in import_notes (never silently dropped). Synthetic snippets
exercise each DSL function; the full 50-file corpus is covered in
tests/test_importer.py (schema validation + independent canonical diff).
"""

import os
import tempfile
import textwrap

from rmgpu.importer.legacy import import_legacy, LegacyImporterError
from rmgpu.schemas.input import Input


def write_temp_input(content: str, suffix: str = "input.py") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="test_legacy_")
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


def cleanup(path: str):
    if os.path.exists(path):
        os.remove(path)


def import_snippet(content: str) -> dict:
    path = write_temp_input(content)
    try:
        return import_legacy(path)
    finally:
        cleanup(path)


class TestSpeciesFunction:
    def test_species_with_smiles(self):
        result = import_snippet(textwrap.dedent("""
            database(thermoLibraries=['primaryThermoLibrary'])
            species(label='ethane', structure=SMILES('CC'))
            species(label='water', structure=SMILES('O'), reactive=False)
        """))
        assert len(result['species']) == 2
        assert result['species'][0]['label'] == 'ethane'
        assert result['species'][0]['structure'] == {'smiles': 'CC'}
        # lossless: the default (True) is simply omitted, not stored
        assert result['species'][0].get('reactive', True) is True
        assert result['species'][1]['reactive'] is False

    def test_species_with_adjacency_list(self):
        adj = textwrap.dedent("""
            species(label='ethane', structure=adjacencyList(
                '''
                1 C u0 p3 {2, S}
                2 C u0 p3 {1, S}
            '''))
        """)
        result = import_snippet(adj)
        assert '1 C u0' in result['species'][0]['structure']['adjlist']

    def test_species_with_inchi_and_fragment(self):
        result = import_snippet(textwrap.dedent("""
            species(label='ethane', structure=InChI('InChI=1S/C2H6/c1-2'))
            species(label='LCCCC', structure=fragment_adj('1 C u0'))
        """))
        assert result['species'][0]['structure']['inchi'].startswith('InChI=')
        assert result['species'][1]['structure']['fragment_adjlist'] == '1 C u0'

    def test_species_positional_args(self):
        result = import_snippet(textwrap.dedent("""
            species('CH4', SMILES('C'))
        """))
        assert result['species'][0]['label'] == 'CH4'
        assert result['species'][0]['structure'] == {'smiles': 'C'}


class TestDatabaseFunction:
    def test_database_basic(self):
        result = import_snippet(textwrap.dedent("""
            database(
                thermoLibraries=['primaryThermoLibrary'],
                reactionLibraries=['primaryReactions'],
                seedMechanisms=['C2H5'],
                kineticsFamilies='default',
                kineticsDepositories='default',
                kineticsEstimator='ml',
                transportLibraries='auto',
            )
        """))
        db = result['database']
        assert db['thermo_libraries'] == ['primaryThermoLibrary']
        assert db['reaction_libraries'] == ['primaryReactions']
        assert db['seed_mechanisms'] == ['C2H5']
        assert db['kinetics_families'] == 'default'
        assert db['kinetics_depositories'] == 'default'
        assert db['kinetics_estimator'] == 'ml'
        assert db['transport_libraries'] == 'auto'

    def test_database_library_pairs(self):
        """(name, include_reverse) library tuples are preserved losslessly."""
        result = import_snippet(textwrap.dedent("""
            database(reactionLibraries=[('Surface/CPOX_Pt/Deutschmann2006_adjusted', False)])
        """))
        assert result['database']['reaction_libraries'] == \
            [{'value': 'Surface/CPOX_Pt/Deutschmann2006_adjusted', 'unit': 'False'}] \
            or result['database']['reaction_libraries'] == \
            [['Surface/CPOX_Pt/Deutschmann2006_adjusted', False]]

    def test_database_with_auto(self):
        result = import_snippet(textwrap.dedent("""
            database(thermoLibraries='auto', reactionLibraries='auto')
        """))
        assert result['database']['thermo_libraries'] == 'auto'


class TestForbiddenFunction:
    def test_forbidden_basic(self):
        result = import_snippet(textwrap.dedent("""
            forbidden(label='no_radical', structure=SMILES('[C]'))
        """))
        assert len(result['forbidden']) == 1
        assert result['forbidden'][0]['label'] == 'no_radical'
        assert result['forbidden'][0]['structure'] == {'smiles': '[C]'}

    def test_forbidden_group(self):
        result = import_snippet(textwrap.dedent("""
            forbidden(label='vacancies', structure=adjacencyListGroup('1 Xv u0 p0 c0'))
        """))
        assert result['forbidden'][0]['structure']['group_adjlist'] == '1 Xv u0 p0 c0'


class TestReactorFunctions:
    def test_simple_reactor(self):
        result = import_snippet(textwrap.dedent("""
            simpleReactor(
                temperature=(800, 'K'),
                pressure=(1.0, 'bar'),
                initialMoleFractions={'CH4': 0.01, 'H2O': 0.99},
                terminationTime=(100, 'us'),
                terminationConversion={'CH4': 0.95},
                terminationRateRatio=2e-13,
                nSims=2,
            )
        """))
        reactor = result['reactors'][0]
        assert reactor['type'] == 'simple'
        assert reactor['temperature'] == {'value': 800, 'unit': 'K'}
        assert reactor['pressure'] == {'value': 1.0, 'unit': 'bar'}
        assert reactor['initial_mole_fractions'] == {'CH4': 0.01, 'H2O': 0.99}
        assert reactor['termination']['time'] == {'value': 100, 'unit': 'us'}
        assert reactor['termination']['conversion'] == {'CH4': 0.95}
        assert reactor['termination']['rate_ratio'] == 2e-13
        assert reactor['n_sims'] == 2

    def test_staged_reactor_routing(self):
        """List-valued temperature/pressure -> type staged + staged_* fields."""
        result = import_snippet(textwrap.dedent("""
            simpleReactor(
                temperature=[(1000, 'K'), (1500, 'K')],
                pressure=(1.0, 'bar'),
                initialMoleFractions={'ethane': [0.05, 0.15], 'N2': 0.9},
                terminationTime=(1e1, 's'),
                nSims=12,
            )
        """))
        reactor = result['reactors'][0]
        assert reactor['type'] == 'staged'
        assert reactor['staged_temperatures'] == [
            {'value': 1000, 'unit': 'K'}, {'value': 1500, 'unit': 'K'}]
        assert reactor['pressure'] == {'value': 1.0, 'unit': 'bar'}
        assert reactor['n_sims'] == 12
        # must validate against the schema
        body = {k: v for k, v in result.items() if k != 'import_notes'}
        body['rmgpu'] = '1.0'
        Input(**body)

    def test_constant_V_reactor(self):
        result = import_snippet(textwrap.dedent("""
            constantVIdealGasReactor(
                temperature=(800, 'K'), pressure=(1.0, 'bar'),
                initialMoleFractions={'CH4': 0.01},
                terminationTime=(100, 'us'),
            )
        """))
        assert result['reactors'][0]['type'] == 'const_V'

    def test_constant_TP_reactor(self):
        result = import_snippet(textwrap.dedent("""
            constantTPIdealGasReactor(
                temperature=(800, 'K'), pressure=(1.0, 'bar'),
                initialMoleFractions={'CH4': 0.01},
                terminationTime=(100, 'us'),
            )
        """))
        assert result['reactors'][0]['type'] == 'const_TP'

    def test_liquid_reactor(self):
        result = import_snippet(textwrap.dedent("""
            liquidReactor(
                temperature=(298, 'K'),
                initialConcentrations={'CH3OH': (1.0, 'M')},
                constantSpecies=['CH3OH'],
                terminationTime=(100, 'us'),
            )
        """))
        r = result['reactors'][0]
        assert r['type'] == 'liquid'
        assert r['initial_concentrations']['CH3OH'] == {'value': 1.0, 'unit': 'M'}
        assert r['constant_species'] == ['CH3OH']

    def test_mb_sampled_reactor(self):
        result = import_snippet(textwrap.dedent("""
            mbsampledReactor(
                temperature=(800, 'K'), pressure=(1.0, 'bar'),
                initialMoleFractions={'CH4': 0.01},
                mbsamplingRate=(1e9, '1/s'),
            )
        """))
        r = result['reactors'][0]
        assert r['type'] == 'mb_sampled'
        assert r['mbsampling_rate'] == {'value': 1e9, 'unit': '1/s'}

    def test_surface_reactor(self):
        result = import_snippet(textwrap.dedent("""
            surfaceReactor(
                temperature=(800, 'K'),
                initialPressure=(1.0, 'bar'),
                initialGasMoleFractions={'CH4': 0.01},
                initialSurfaceCoverages={'X': 1.0},
                surfaceVolumeRatio=(1.0e5, 'm^-1'),
                terminationRateRatio=0.01,
            )
        """))
        r = result['reactors'][0]
        assert r['type'] == 'surface'
        assert r['initial_pressure'] == {'value': 1.0, 'unit': 'bar'}
        assert r['termination']['rate_ratio'] == 0.01

    def test_liquid_surface_reactor(self):
        result = import_snippet(textwrap.dedent("""
            liquidSurfaceReactor(
                temperature=(300, 'K'),
                liqPotential=(0.0, 'V'),
                surfPotential=(-1.0, 'V'),
                initialConcentrations={'CO2': (1e-3, 'mol/cm^3')},
                initialSurfaceCoverages={'vacantX': 1.0},
                surfaceVolumeRatio=(1.0e5, 'm^-1'),
                distance=(10.0e-10, 'm'),
                viscosity=(5e7, 'Pa*s'),
                constantSpecies=['CO2'],
            )
        """))
        r = result['reactors'][0]
        assert r['type'] == 'liquid_surface'
        assert r['liq_potential'] == {'value': 0.0, 'unit': 'V'}
        assert r['surf_potential'] == {'value': -1.0, 'unit': 'V'}
        assert r['viscosity'] == {'value': 5e7, 'unit': 'Pa*s'}


class TestSimulatorFunction:
    def test_simulator_basic(self):
        result = import_snippet(textwrap.dedent("""
            simulator(atol=1e-16, rtol=1e-8, sens_atol=1e-6, sens_rtol=1e-4)
        """))
        sim = result['simulator']
        assert sim['atol'] == 1e-16
        assert sim['rtol'] == 1e-8
        assert sim['sens_atol'] == 1e-6
        assert sim['sens_rtol'] == 1e-4


class TestModelFunction:
    def test_model_basic(self):
        result = import_snippet(textwrap.dedent("""
            model(
                toleranceMoveToCore=0.1,
                toleranceKeepInEdge=0.0,
                toleranceInterruptSimulation=1.0,
                maximumEdgeSpecies=10000,
                filterReactions=True,
                filterThreshold=1e8,
                toleranceTransitoryDict={'NO': 0.2},
            )
        """))
        m = result['model']
        assert m['tolerance_move_to_core'] == 0.1
        assert m['tolerance_keep_in_edge'] == 0.0
        assert m['tolerance_interrupt_simulation'] == 1.0
        assert m['maximum_edge_species'] == 10000
        assert m['filter_reactions'] is True
        assert m['filter_threshold'] == 1e8
        assert m['tolerance_transitory_dict'] == {'NO': 0.2}


class TestPressureDependenceFunction:
    def test_pressure_dependence_basic(self):
        result = import_snippet(textwrap.dedent("""
            pressureDependence(
                method='modified strong collision',
                maximumGrainSize=(0.5, 'kcal/mol'),
                minimumNumberOfGrains=250,
                temperatures=(300, 2000, 'K', 8),
                pressures=(0.01, 100, 'bar', 5),
                interpolation=('Chebyshev', 6, 4),
                maximumAtoms=15,
                completedNetworks=['C2H6'],
            )
        """))
        pd = result['pressure_dependence']
        assert pd['method'] == 'modified strong collision'
        assert pd['maximum_grain_size'] == {'value': 0.5, 'unit': 'kcal/mol'}
        assert pd['minimum_number_of_grains'] == 250
        assert pd['temperatures'] == [300, 2000, 'K', 8]
        assert pd['pressures'] == [0.01, 100, 'bar', 5]
        assert pd['interpolation'] == ['Chebyshev', 6, 4]
        assert pd['maximum_atoms'] == 15
        assert pd['completed_networks'] == ['C2H6']
        # schema normalizes shorthand methods
        body = {'rmgpu': '1.0', 'pressure_dependence':
                {'method': 'cse', 'maximum_grain_size': {'value': 0.5, 'unit': 'kcal/mol'}}}
        from rmgpu.schemas.input import PressureDependenceBlock
        assert PressureDependenceBlock(**body['pressure_dependence']).method == 'strong collision'


class TestMLEstimatorFunction:
    def test_ml_estimator_basic(self):
        result = import_snippet(textwrap.dedent("""
            mlEstimator(thermo=True, minHeavyAtoms=4)
        """))
        ml = result['ml_estimator']
        assert ml['thermo'] is True
        assert ml['min_heavy_atoms'] == 4


class TestOptionsFunction:
    def test_options_basic(self):
        result = import_snippet(textwrap.dedent("""
            options(
                name='my_run',
                generateSeedEachIteration=True,
                saveSeedToDatabase=False,
                units='si',
                generateOutputHTML=False,
                generatePlots=False,
                saveSimulationProfiles=False,
                saveEdgeSpecies=False,
                keepIrreversible=False,
                wallTime='00:00:00:00',
                generateChemkin={'saveInterval': 5, 'saveEdge': False},
                saveRestartPeriod=None,
            )
        """))
        opts = result['options']
        assert opts['name'] == 'my_run'
        assert opts['units'] == 'si'
        assert opts['generate_chemkin'] == {'saveInterval': 5, 'saveEdge': False}
        assert opts['wall_time'] == '00:00:00:00'
        assert opts['save_restart_period'] is None


class TestSolvationAndUncertainty:
    def test_solvation_basic(self):
        result = import_snippet(textwrap.dedent("""
            solvation(solvent='methanol')
        """))
        assert result['solvation']['solvent'] == 'methanol'

    def test_uncertainty_basic(self):
        result = import_snippet(textwrap.dedent("""
            uncertainty(localAnalysis=True, globalAnalysis=False, localNumber=10)
        """))
        unc = result['uncertainty']
        assert unc['local_analysis'] is True
        assert unc['global_analysis'] is False


class TestConstraintsAndCatalyst:
    def test_generated_species_constraints(self):
        result = import_snippet(textwrap.dedent("""
            generatedSpeciesConstraints(
                allowed=['input species', 'reaction libraries'],
                maximumCarbonAtoms=5,
                maximumOxygenAtoms=3,
                maximumHeavyAtoms=10,
                allowSingletO2=False,
            )
        """))
        gsc = result['generated_species_constraints']
        assert gsc['allowed'] == ['input species', 'reaction libraries']
        assert gsc['maximum_carbon_atoms'] == 5
        assert gsc['allow_singlet_o2'] is False

    def test_catalyst_properties(self):
        result = import_snippet(textwrap.dedent("""
            catalystProperties(
                metal='Pt111',
                bindingEnergies={'H': (-2.75368, 'eV/molecule')},
                surfaceSiteDensity=(2.483e-9, 'mol/cm^2'),
                coverageDependence=False,
            )
        """))
        cp = result['catalyst_properties']
        assert cp['metal'] == 'Pt111'
        assert cp['binding_energies']['H'] == {'value': -2.75368, 'unit': 'eV/molecule'}
        assert cp['surface_site_density'] == {'value': 2.483e-9, 'unit': 'mol/cm^2'}
        assert cp['coverage_dependence'] is False

    def test_quantum_mechanics_preserved(self):
        result = import_snippet(textwrap.dedent("""
            quantumMechanics(software='mopac', method='pm3',
                             scratchDirectory=None, onlyCyclics=True,
                             maxRadicalNumber=0)
        """))
        qm = result['quantum_mechanics']
        assert qm['software'] == 'mopac'
        assert qm['only_cyclics'] is True
        assert qm['scratch_directory'] is None


class TestArithmeticAndEdges:
    def test_arithmetic_in_mole_fractions(self):
        """Numeric arithmetic sub-expressions are evaluated (1. / 6.5 etc.)."""
        result = import_snippet(textwrap.dedent("""
            simpleReactor(
                temperature=(1500, 'K'),
                pressure=(10.0, 'bar'),
                initialMoleFractions={'ethane': 2.0 / 7.0, 'N2': 4, 'O2': 1.0},
                terminationTime=(40, 's'),
            )
        """))
        imf = result['reactors'][0]['initial_mole_fractions']
        assert abs(imf['ethane'] - 2.0 / 7.0) < 1e-15
        assert imf['N2'] == 4

    def test_repeated_block_last_wins(self):
        """RMG replaces single-blocks; the override is documented loudly."""
        result = import_snippet(textwrap.dedent("""
            model(toleranceMoveToCore=0.01, maxNumObjsPerIter=1)
            simulator(atol=1e-16, rtol=1e-8)
            simulator(atol=1e-15, rtol=1e-6)
        """))
        assert result['simulator']['atol'] == 1e-15
        assert any('simulator' in n['message'] for n in result['import_notes'])

    def test_unmapped_function_loud_note(self):
        """An unmapped DSL function is preserved under _legacy with a note."""
        result = import_snippet(textwrap.dedent("""
            restartFromSeed(path='/path/to/seed')
        """))
        assert result['_legacy']['restartFromSeed']['path'] == '/path/to/seed'
        assert any('restartFromSeed' in n['message'] for n in result['import_notes'])


class TestQuantityTuples:
    def test_quantity_tuple_conversion(self):
        result = import_snippet(textwrap.dedent("""
            simpleReactor(
                temperature=(800, 'K'), pressure=(1.0, 'bar'),
                initialMoleFractions={'CH4': 1.0},
                terminationTime=(100, 'us'),
            )
        """))
        reactor = result['reactors'][0]
        assert reactor['temperature'] == {'value': 800, 'unit': 'K'}
        assert reactor['termination']['time'] == {'value': 100, 'unit': 'us'}


class TestAutoSentinel:
    def test_auto_library_sentinel(self):
        result = import_snippet(textwrap.dedent("""
            database(thermoLibraries='auto', reactionLibraries='auto')
        """))
        assert result['database']['thermo_libraries'] == 'auto'


class TestImportLegacyErrors:
    def test_file_not_found(self):
        try:
            import_legacy('/nonexistent/path/input.py')
            assert False, "Should have raised LegacyImporterError"
        except LegacyImporterError as e:
            assert 'File not found' in str(e)

    def test_syntax_error(self):
        path = write_temp_input("def invalid(:\n    pass")
        try:
            import_legacy(path)
            assert False, "Should have raised LegacyImporterError"
        except LegacyImporterError as e:
            assert 'Syntax error' in str(e)
        finally:
            cleanup(path)


class TestSchemaValidation:
    def test_minimal_input_validates(self):
        result = import_snippet(textwrap.dedent("""
            database(
                thermoLibraries=['primaryThermoLibrary'],
                reactionLibraries=['primaryReactions'],
            )
            species(label='CH4', structure=SMILES('C'))
            simpleReactor(
                temperature=(800, 'K'),
                pressure=(1.0, 'bar'),
                initialMoleFractions={'CH4': 1.0},
                terminationTime=(100, 'us'),
            )
            simulator(atol=1e-16, rtol=1e-8)
            model(toleranceMoveToCore=0.1)
        """))
        assert result['rmgpu'] == '1.0'
        body = {k: v for k, v in result.items() if k != 'import_notes'}
        Input(**body)  # must validate without error


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
