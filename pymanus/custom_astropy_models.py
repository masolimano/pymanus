import numpy as np
import astropy.modeling as am
FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
GAUSSIAN_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

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
        return self.width * GAUSSIAN_SIGMA_TO_FWHM / (1 - 2 * np.log(2) * asym ** 2)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    wav = np.linspace(-2000, 2000, 300)
    profile = AsymmetricGaussian1D(x_peak=0, asym=0.2, width=90, amplitude=1)
    sampled_profile = profile(wav)
    imin, zero = np.argmin(sampled_profile), np.min(sampled_profile)
    sampled_profile[:imin] = zero
    plt.plot(wav, sampled_profile)
    plt.show()
