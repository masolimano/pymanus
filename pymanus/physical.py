import numpy as np
import astropy.units as u
import uncertainties as unc
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import BlackBody
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def luminosity_prime(obs_flux, redshift, obs_freq):
    """
    Computes the *primed* line luminosity as definied by
    Carilli & Walter (2013).

    Parameters
    ----------
    obs_flux: ufloat
        Observed velocity-integrated flux in Jy.km/s
    redshift: float
        Redshift of source-plane
    obs_freq: float
        Observed frequency in GHz

    Returns
    ------
    lum_prime: ufloat
         in units of K km/s pc^(-2)
    """

    lum_distance = cosmo.luminosity_distance(redshift).to(u.Mpc).value
    cosmo_factor = lum_distance ** 2 / ((1 + redshift) ** 3 * (obs_freq ** 2))
    return 3.25e7 * obs_flux * cosmo_factor
