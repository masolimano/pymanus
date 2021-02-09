import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import BlackBody
from scipy.constants import h, k # These are not in cgs but doesn't matter since we use ratios anyway
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

def gamma_rj(t_d, nu_obs, z):
    """
    Rayleigh-Jeans correction factor. Frequencies (Hz) are given in observed frame.
    """
    beta = h * nu_obs * (1 + z) / (k * t_d)
    return beta / (np.exp(beta) - 1)

def dust_mass_following_casey19(
    nu_obs,
    s_obs,
    z,
    nu_ref=6.662e11,
    kappa_ref=1.3,
    beta=1.8,
    t_dust=25,
    correct_cmb=True
):
    """
    Compute the total dust mass from observed far-IR flux according to the
    formula given in Casey et al. (2019).

    Parameters
    ---------
    nu_obs: float
        Observed frequency in Hertz
    s_obs : (u)float
        Observed flux at the observed frequency in mJy
    z: float
        Redshift of the source
    nu_ref: float
        Reference frequency (rest frame) in Hertz
    kappa_ref: float
        Dust mass absorption coefficient (opacity) at `nu_ref` in cm^2 g^-1
    beta: float
        Dust emissivity index
    t_dust:
        Dust temperature in Kelvin
    correct_cmb:
        Account for the effect of CMB heating (relevant at z>3-4)

    Return
    --------
    mdust: ufloat
        Total dust mass in units of Solar Mass
    """
    dust_bb = BlackBody(temperature=t_dust * u.K)
    cmb_bb = BlackBody(temperature=cosmo.Tcmb(z))

    # redshifting
    # nu_rest = nu_obs / (1 + z)   <--- This is wrong!
    nu_rest = nu_obs * (1 + z)

    # Flux to luminosity
    factor1 = s_obs * cosmo.luminosity_distance(z).to(u.cm).value ** 2 * (1 + z) ** (-3 - beta) / \
        (kappa_ref * dust_bb(nu_ref * u.Hz).value)


    # Dust opacity scaling
    factor2 = (nu_ref / nu_obs) ** (2 + beta)

    # Rayleigh-Jeans tail scaling
    factor3 = gamma_rj(t_dust, nu_ref, 0) / gamma_rj(t_dust, nu_obs, z)

    # CMB heating correction factor
    if correct_cmb:
        factor4 = 1 / (1 - cmb_bb(nu_rest * u.Hz).value / dust_bb(nu_rest * u.Hz).value)
        return factor1 * factor2 * factor3 * factor4 / 1.988e33
    else:
        return factor1 * factor2 * factor3 / 1.988e33
