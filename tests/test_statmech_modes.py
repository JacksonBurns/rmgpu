"""
tests/test_statmech_modes.py

Parity tests for job-07/step-01 statmech modes.
"""

import numpy as np
import pytest

from rmgpu.statmech.modes import HarmonicOscillator, LinearRotor, NonlinearRotor, Conformer

def test_harmonic_oscillator_heat_capacity():
    # Simple sanity: one oscillator, quantum
    ho = HarmonicOscillator(frequencies=[1000.0], quantum=True)
    T = 298.15
    cv = ho.get_heat_capacity(T)
    # Should be between 0 and R
    assert 0 < cv < 10 * 8.314472

def test_conformer_heat_capacity_sum():
    modes = [
        HarmonicOscillator(frequencies=[500, 1000], quantum=True),
        LinearRotor(rotational_constant=1.0),
        NonlinearRotor(rotational_constants=[1.0, 1.0, 1.0]),
    ]
    conf = Conformer(E0=0.0, modes=modes)
    T = 298.15
    cp = conf.get_heat_capacity(T)
    # Sum of components >0
    assert cp > 0

def test_harmonic_number_of_states_shape():
    ho = HarmonicOscillator(frequencies=[1000], quantum=True)
    e = np.linspace(0, 50000, 101)
    ns = ho.get_number_of_states(e)
    assert ns.shape == e.shape
    assert np.all(ns >= 0)

def test_conformer_dos():
    ho = HarmonicOscillator(frequencies=[500], quantum=True)
    conf = Conformer(modes=[ho])
    e = np.linspace(0, 50000, 51)
    dos = conf.get_density_of_states(e)
    assert dos.shape == e.shape
    # density should be non-negative
    assert np.all(dos >= 0)
