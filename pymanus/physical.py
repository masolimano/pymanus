import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import BlackBody
#from scipy.constants import h, k # These are not in cgs but doesn't matter since we use ratios anyway
from astropy.constants import h, k_B
from scipy.special import gammaincinv
import uncertainties as unc
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

def gamma_rj(nu_obs, z, t_d):
    """
    Rayleigh-Jeans correction factor. Frequencies (Hz) are given in observed frame.
    Equation (6) from  Scoville et al. 2016, ApJ, 820, 83.

    Parameters
    ---------
    t_d: float
        Dust temperature in Kelvin
    nu_obs: float
        Observed frequency in Hertz
    z: float
        Redshift
    """
    beta = h * nu_obs * (1 + z) / (k_B * t_d)
    return beta / (np.exp(beta) - 1)

def luminosity_nu_rest(s_obs, z):
    """
    Rest-frame specific luminosity.

    Parameters
    ----------
    s_obs: Quantity
        Flux density in the F_nu scheme.
    z: float
        Redshift.
    """
    luminosity_distance = cosmo.luminosity_distance(z).to(u.Mpc)
    return s_obs * 4 * np.pi * luminosity_distance ** 2 / (1 + z)

def luminosity_nu_850um(s_obs, z, nu_obs, t_d, beta):
    """
    850µm specific luminosity assuming Rayleigh-Jeans far-ir spectrum,
    dust temperature, and dust emissivity.

    Parameters
    ----------
    s_obs: Quantity
        Observed flux density in F_nu scheme.
    z: float
        Redshift.
    nu_obs: Quantity
        Observed frequency.
    t_d: Quantity
        Dust temperature.
    beta: float
        Dust emissivity.
    """
    nu_850 = (850 * u.um).to(u.GHz, equivalencies=u.spectral())
    nu_rest = nu_obs * (1 + z)
    rj_correction = gamma_rj(nu_850, 0, t_d) / gamma_rj(nu_obs, z, t_d)
    return luminosity_nu_rest(s_obs, z) * rj_correction * (nu_850 / nu_rest) ** (2 + beta)

def integrated_flux_from_Lline(Lline, nu_obs, z):
    """
    Compute velocity-integrated flux density from
    L'line luminosity.

    Lline: Quantity
        L'line luminosity, typically in units of K km/s pc^2
    nu_obs: Quantity
        Line observed-frame frequency
    z: float
        Redshift.
    """
    const = 3.25e-5 * u.K * u.GHz ** 2 / u. Jy
    lum_distance2 = cosmo.luminosity_distance(z) ** 2
    return (Lline * (1 + z) ** 3 * nu_obs ** 2 / lum_distance2 / const).to(u.Jy * u.km / u.s)

def atomic_carbon_mass_following_Weiss05(t_ex, LprimeCI):
    """
    Atomic carbon mass from the [C I](1-0) luminosity following
    Weiss et al. (2005).

    t_ex: (u)float
        Excitation temperature of [C I]
    LprimeCI: (u) float
        Observed prime-Luminosity of the [CI](1-0)
        line in units of K km/s pc^2
    """
    Q = 1 + 3 * umath.exp(- 23.6 / t_ex) + 5 * umath.exp(- 62.5 / t_ex)
    M_CI = 5.706e-4 * Q * umath.exp(23.6 / t_ex) * LprimeCI / 3
    return M_CI

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
        Account for the effect of CMB heating (relevant at z > 4)

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
    s_obs = (s_obs * u.mJy).to(u.erg/u.s/u.cm**2/u.Hz).value

    # Flux to luminosity
    factor1 = s_obs * cosmo.luminosity_distance(z).to(u.cm).value ** 2 * (1 + z) ** (-3 - beta) / \
        (kappa_ref * (dust_bb(nu_ref * u.Hz) * u.sr).to(u.erg/u.s/u.cm**2/u.Hz).value)


    # Dust opacity scaling
    factor2 = (nu_ref / nu_obs) ** (2 + beta)

    # Rayleigh-Jeans tail scaling
    factor3 = (gamma_rj(nu_ref * u.Hz, 0, t_d=t_dust * u.K) / gamma_rj(nu_obs * u.Hz, z, t_dust * u.K)).value

    # CMB heating correction factor
    if correct_cmb:
        factor4 = 1 / (1 - cmb_bb(nu_rest * u.Hz).value / dust_bb(nu_rest * u.Hz).value)
        return factor1 * factor2 * factor3 * factor4 / 1.988e33
    else:
        return factor1 * factor2 * factor3 / 1.988e33

@unc.wrap
def sersic_percentile_radius(n, q):
    """
    Calculates the q *100 percent luminosity radius of a Sérsic profile of index n.
    Reference: https://ui.adsabs.harvard.edu/abs/1999A%26A...352..447C/abstract

    Parameters
    ----------
    n: float
        Sérsic index
    q: float
        Fraction(between 0.0 and 1.0) of the total luminosity.
    """
    bn = gammaincinv(2 * n, 0.5)
    eta = (gammaincinv(2 * n, q) / bn) ** n
    return eta

if __name__ == '__main__':
    Snu = 4.0 # mJy
    nu_obs = 3.435e11
    z = 2.3899
    mdust = dust_mass_following_casey19(nu_obs, Snu, z, beta=1.8, correct_cmb=False)
    print(f'{mdust:.2e}')
