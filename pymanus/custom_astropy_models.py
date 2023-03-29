import numpy as np
import astropy.modeling as am
from astropy.modeling.fitting import (_validate_model,
                                      _fitter_to_model_params,
                                      model_to_fit_params, Fitter,
                                      _convert_input)
from astropy.modeling.optimizers import Simplex
from astropy.convolution import convolve, convolve_fft
FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
GAUSSIAN_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def chi2_1d_with_lsf(measured_vals, updated_model, lsf, yerr, x):
    """
    Custom likelihood for accounting for the instrument LSF when fitting spectral lines
    """
    model_vals = updated_model(x)
    model_vals_convolved = convolve(model_vals, lsf, preserve_nan=True)
    if yerr is None:
        return np.nansum((measured_vals - model_vals_convolved) ** 2)
    else:
        return np.nansum((measured_vals - model_vals_convolved) ** 2 / yerr ** 2)

class Fitter1DLSF(Fitter):

    def __init__(self, optimizer=Simplex):
        self.statistic = chi2_1d_with_lsf
        super().__init__(optimizer, statistic=self.statistic)

    def __call__(self, model, x, ydata, lsf, yerr=None, **kwargs):
        model_copy = _validate_model(model,
                                     self._opt_method.supported_constraints)
        farg = _convert_input(x, ydata)
        farg = (model_copy, lsf, yerr) + farg
        p0, _, _ = model_to_fit_params(model_copy)

        fitparams, self.fit_info = self._opt_method(
            self.objective_function, p0, farg, **kwargs
        )
        _fitter_to_model_params(model_copy, fitparams)
        return model_copy


class AsymmetricGaussian1D(am.Fittable1DModel):
    """
    Simple asymmetric Gaussian profile formulated by Shibuya+14.
    Implementation is based off ~astropy.modeling.models.Gaussian1D source code.
    """
    x_peak = am.Parameter(default=0)
    asym = am.Parameter(default=1)

    # Ensure width makes sense if its bounds are not explicitly set.
    # width must be non-zero and positive.
    #width = am.Parameter(default=1, bounds=(FLOAT_EPSILON, None))
    width = am.Parameter(default=1)
    amplitude = am.Parameter(default=1)

    @staticmethod
    def evaluate(x, x_peak, asym, width, amplitude):
        delta_x = x - x_peak
        return amplitude * np.exp(- 0.5 * delta_x ** 2 / (asym * delta_x + width) ** 2)

    @staticmethod
    def fit_deriv(x, x_peak, asym, width, amplitude):
        delta_x = x - x_peak
        adxd = asym * delta_x + width
        evx = amplitude * np.exp(-0.5 * delta_x ** 2 / adxd ** 2)

        d_x_peak = evx * delta_x * (adxd - delta_x * asym) / adxd ** 3
        d_amplitude = evx / amplitude
        d_width = evx * (delta_x ** 2 / adxd ** 3)
        d_asym = evx * (delta_x / adxd) ** 3
        return [d_x_peak, d_asym, d_width, d_amplitude]


    @property
    def fwhm(self):
        """
        Full width at half maximum
        """
        return self.width * GAUSSIAN_SIGMA_TO_FWHM / (1 - 2 * np.log(2) * self.asym ** 2)

class PiecewiseLinear1D(am.Fittable1DModel):
    """
    Piecewise function that is flat zero up to certain value of x
    and then turns into a linear function. Allows for discontinuity
    at x_0 (i.e is not a ReLU but a regular line convolved with a step function).
    Can be useful to model a faint continuum redwards of Lya or the Lyman break in
    galaxy spectra.
    """
    x_0 = am.Parameter(default=0)
    slope = am.Parameter(default=1)
    intercept = am.Parameter(default=0)

    @staticmethod
    def evaluate(x, x_0, slope, intercept):
        line = lambda w: slope * w + intercept
        zero = lambda w: 0
        return np.piecewise(x, [x < x_0, x>= x_0], [zero, line])

    @staticmethod
    def fit_deriv(x, x_0, slope, intercept):
        idd = lambda w: w
        zero = lambda w: 0
        one = lambda w: 1
        d_x_0 = np.zeros_like(x)
        d_slope = np.piecewise(x, [x < x_0, x >= x_0], [zero, idd])
        d_intercept = np.piecewise(x, [x < x_0, x >= x_0], [zero, one])
        return [d_x_0, d_slope, d_intercept]


class DiracDelta2D(am.Fittable2DModel):
    """
    2D Dirac Delta model, meant to represent point-like astronomical sources
    in any gridded numerical representation.
    """
    amplitude = am.Parameter()
    x_0 = am.Parameter()
    y_0 = am.Parameter()

    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0):
        delta = np.zeros_like(x)
        x1d = x[0]
        y1d = y[:, 0]
        i = np.argmin(np.abs(x1d - x_0))
        j = np.argmin(np.abs(y1d - y_0))
        delta[j, i] = amplitude
        return delta

    @staticmethod
    def fit_deriv(x, y, amplitude, x_0, y_0):
        return None

class TruncatedExp1D(am.Fittable1DModel):
    """
    Truncated exponential profile. Might be useful to model asymmetric lines.
    """
    amplitude = am.Parameter()
    x_peak = am.Parameter()
    tau = am.Parameter()

    @staticmethod
    def evaluate(x, amplitude, x_peak, tau):
        delta_x = x - x_peak
        eval_range = delta_x * np.sign(tau) >= 0
        zero_range = np.logical_not(eval_range)
        result = np.select([eval_range, zero_range],
                           [amplitude * np.exp(-delta_x / tau), 0])
        return result

    @staticmethod
    def fit_deriv(x, amplitude, x_peak, tau):
        delta_x = x - x_peak
        eval_range = delta_x * np.sign(tau) >= 0
        zero_range = np.logical_not(eval_range)
        result = np.select([eval_range, zero_range],
                           [(-1 / tau) * amplitude * np.exp(-delta_x / tau), 0])
        return result
    @property

    def fwhm(self):
        """
        Full width at half maximum
        """
        return np.abs(self.tau) * np.log(2)

class SplitGaussian1D(am.Fittable1DModel):
    """
    Alternative implementation of asymmetric Gaussian. This time is just
    two Gaussians with different widths (but same amplitude) stitched together
    at a given value of x.
    """
    amplitude = am.Parameter()
    center = am.Parameter()
    log_ratio = am.Parameter()
    fwhm = am.Parameter()

    @staticmethod
    def evaluate(x, amplitude, center, log_ratio, fwhm):
        x_prime = x - center
        r = 10 ** log_ratio
        sigma = fwhm / GAUSSIAN_SIGMA_TO_FWHM
        sigma_minus = sigma / (1 + r)
        sigma_plus = sigma * r / (1 + r)
        plus_range = x_prime >= 0
        minus_range = np.logical_not(plus_range)

        gauss_minus = amplitude * np.exp( -0.5 * x_prime ** 2 / sigma_minus ** 2)
        gauss_plus = amplitude * np.exp( -0.5 * x_prime ** 2 / sigma_plus ** 2)

        return np.select([minus_range, plus_range], [gauss_minus, gauss_plus])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    wav = np.linspace(0, 100, 501)
    profile_red = TruncatedExp1D(x_peak=43, amplitude=1.2, tau=2.5)
    profile_blue = TruncatedExp1D(x_peak=39, amplitude=.8, tau=-5)
    ydata = (profile_red + profile_blue)(wav)
    lsf = am.models.Gaussian1D(mean=50, stddev=1.5)(wav)
    ydata_conv = convolve_fft(ydata, lsf)
    ydata_conv_noise = ydata_conv + np.random.normal(loc=0, scale=0.05, size=ydata.size)

    init = TruncatedExp1D(x_peak=44, amplitude=1, tau=3) + TruncatedExp1D(x_peak=38, amplitude=.7, tau=-4)
    fit = Fitter1DLSF()
    best = fit(init, wav, ydata_conv_noise, lsf, yerr=0.05, maxiter=9000)



    plt.plot(wav, ydata_conv_noise)
    plt.plot(wav, ydata)
    plt.plot(wav, lsf)
    plt.plot(wav, best(wav))
    plt.show()


