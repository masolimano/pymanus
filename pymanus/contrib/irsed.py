"""
Utilities to compute and convert modified blackbody
SEDs and related quantities. Most of these functions were taken
from Patrick Drew's mcirsed package.

https://github.com/pdrew32/mcirsed

(c) Manuel Solimano 2023
(c) Patrick Drew 2022
"""
import numpy as np
from astropy import units as u
from astropy import constants as c
from scipy.special import expit

hck = (c.h*c.c/c.k_B).to(u.micron*u.K).value
xHz = np.linspace((c.c/(8.*u.micron)).decompose().value,
                      (c.c/(1000.*u.micron)).decompose().value, 100000)[::-1]
xWa = (c.c/xHz/u.Hz).decompose().to(u.um)[::-1].value
deltaHz = xHz[1]-xHz[0]

def BB(nbb, Tdust, beta, w0, restWave):
    """Modified blackbody function

    Math takes the form:
    10**nbb * (1.0-np.exp(-(w0/restWave)**beta)) * restWave**(-3.0) / (np.exp(ah.h.hck/restWave/Tdust)-1.0)
    """
    a = np.power(10, nbb)
    b = np.subtract(1.0, np.exp(-np.power(np.true_divide(w0, restWave), beta)))
    c = np.power(restWave, -3.0)
    d = np.subtract(np.exp(np.true_divide(np.true_divide(hck, restWave), Tdust)), 1.0)
    return np.true_divide(np.multiply(np.multiply(a, b), c), d)


def powerLaw(npl, restWave, alpha):
    """Equation of the power law portion of SED"""
    return np.multiply(npl, np.power(restWave, alpha))


def derivativeLogBB(Tdust, beta, w0):
    """Solves for the (approx) derivatives of the BB function."""
    extra_fine_rest_wave = np.logspace(np.log10(20), np.log10(200), 1000)
    log_bb = np.log10(BB(10.0, Tdust, beta, w0, extra_fine_rest_wave))
    delta_y = log_bb[1:] - log_bb[:-1]
    delta_x = np.log10(extra_fine_rest_wave[1:]) - np.log10(extra_fine_rest_wave[:-1])
    return delta_y / delta_x


def eqWave(alpha, Tdust, beta, w0):
    """Compute the wavelength where the derivative of the log of BB equals the slope of the power law"""
    der_bb_reverse = derivativeLogBB(Tdust, beta, w0)[::-1]
    # only search 20um to 200um because the eqWave is definitely between there
    extra_fine_rest_wave = np.logspace(np.log10(20), np.log10(200), 1000)[::-1]
    return extra_fine_rest_wave[np.searchsorted(der_bb_reverse, alpha)]


def SnuNoBump(norm1, Tdust, alpha, beta, w0, restWave):
    """Combined MBB and Power Law functional form to fit with MCMC
    For speed of computation, uses a sigmoid function with a sharp cutoff
    times BB and 1-sigmoid times pl to achieve the piecewise function.
    """
    eq_w = eqWave(alpha, Tdust, beta, w0)
    bb = BB(norm1, Tdust, beta, w0, restWave)
    n = BB(norm1, Tdust, beta, w0, eq_w) * eq_w**-alpha
    pl = powerLaw(n, restWave, alpha)
    sig = expit(200*(restWave-eq_w))
    return (1-sig) * pl + sig * bb


def SnuCasey2012(norm1, Tdust, alpha, beta, w0, restWave):
    """
    MBB+powerlaw as defined in Casey (2012, see their Table 1)
    WARNING: I do not know if the b coefficients will work for
    a w0 different from 200 µm.
    """
    b1, b2, b3, b4 = 28.68, 6.246, 1.905e-4, 7.243e-5
    bb = BB(norm1, Tdust, beta, w0, restWave)
    eq_w = 1 / (np.power(b1 + b2 * alpha, -2) + (b3 + b4 * alpha) * Tdust)
    norm_pl = BB(norm1, Tdust, beta, w0, eq_w) * np.power(eq_w, -alpha)
    pl = powerLaw(norm_pl, restWave, alpha) * np.exp(-(restWave / eq_w) ** 2)
    return pl + bb


def IRLum(norm1, Tdust, alpha, beta, w0, z, fourPiLumDistSquared):
    """
    Calculate LIR. Output is in dex(Lsun) units

    norm1: float
        logarithmic normalization factor. Something close to 10 will put the
        peak near 1000 mJy (assuming w0 = 200 microns)

    Tdust: float
        Dust temperature in Kelvin

    alpha: float
        Mid-IR powerlaw index
    """
    conversionFactor = 2.4873056783618645e-11  # mJy Hz Mpc^2 to lsol
    return np.log10(np.sum(SnuNoBump(norm1, Tdust, alpha, beta, w0, xWa)) * deltaHz/(1+z) * fourPiLumDistSquared) + np.log10(conversionFactor)


def nuLnu_to_LIR_ratio_MBB(Tdust, beta, w0, restWave):
    nu = (restWave * u.um).to(u.Hz, equivalencies=u.spectral()).value
    nbb = 10 # does not matter bc it cancels out
    return nu * BB(10, Tdust, beta, w0, restWave) / (np.sum(BB(10, Tdust, beta, w0, xWa)) * deltaHz)

def nuLnu_to_LIR_ratio(Tdust, alpha, beta, w0, restWave):
    nu = (restWave * u.um).to(u.Hz, equivalencies=u.spectral()).value
    nbb = 10 # does not matter bc it cancels out
    return nu * SnuNoBump(10, Tdust, alpha, beta, w0, restWave) / (np.sum(SnuNoBump(10, Tdust, alpha, beta, w0, xWa)) * deltaHz)

def lambdaPeak(norm1, Tdust, alpha, beta, w0):
    """Calculate Peak Wavelength"""
    x = xWa.astype('float64')
    return x[np.argmax(SnuNoBump(norm1, Tdust, alpha, beta, w0, x))]


def Tredshift0(redshift, beta, Tdust):
    """equation for calculating the dust temperature if the galaxy were at z=0"""
    power = 4+beta
    return (Tdust**power - cosmo.Tcmb0.value**power * ((1+redshift)**power - 1)) ** (1/power)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    sed  = SnuCasey2012(10.55, Tdust=22.9, alpha=1.9, beta=1.8, w0=200, restWave=xWa)
    plt.loglog(xWa, sed)
    plt.scatter([24],[5.5])
    plt.ylim(1, 1e3)
    plt.show()
