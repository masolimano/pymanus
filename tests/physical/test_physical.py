from ...pymanus.physical import luminosity_prime
import numpy as np
from uncertainties import ufloat

class TestLuminosityPrime:
    def test_aztec2_float_co54A(self):
        """
        Testing `luminosity_prime` for non-ufloat input against
        ^12CO(5->4) luminosity of AzTEC2-A according to Jiménez-Andrade et al. (2020)
        """
        obs_flux = 1.34  # Jy km / s
        redshift = 4.626
        obs_freq = 102.43 # GHz
        expected_outcome = 4.1e10 # K.km/s.pc^2
        lum_prime = luminosity_prime(obs_flux, redshift, obs_freq)
        assert abs(expected_outcome - lum_prime) < 1e10

    def test_aztec2_float_co54B(self):
        """
        Testing `luminosity_prime` for non-ufloat input against
        ^12CO(5->4) luminosity of AzTEC2-B according to Jiménez-Andrade et al. (2020)
        """
        obs_flux = 0.325  # Jy km / s
        redshift = 4.633
        obs_freq = 102.30 # GHz
        expected_outcome = 1.0e10 # K.km/s.pc^2
        lum_prime = luminosity_prime(obs_flux, redshift, obs_freq)
        assert abs(expected_outcome - lum_prime) < 1e9

    def test_aztec2_ufloat_co54A(self):
        """
        Testing `luminosity_prime` for ufloat input against
        ^12CO(5->4) luminosity of AzTEC2-A according to Jiménez-Andrade et al. (2020)
        """
        obs_flux = ufloat(1.34, 0.1)  # Jy km / s
        redshift = 4.626
        obs_freq = 102.43 # GHz
        expected_outcome = 4.1e10 # K.km/s.pc^2
        expected_error = 0.2e10
        lum_prime = luminosity_prime(obs_flux, redshift, obs_freq)
        assert abs(expected_outcome - lum_prime.nominal_value) < expected_error
        assert abs(expected_error - lum_prime.std_dev) < 1e10
