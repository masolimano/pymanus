"""
Sub module for non trivial unit conversions

(c) M. Solimano 2020
"""


def air2vac(wav_air):
    """
    Air to vacuum wavelength conversion according to N. Piskunov.
    See https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    NOTE: This is valid only for wavelengths greater than 2000AA!

    Parmeters
    --------
    wav_air: array
        Air wavelengths in units of Angstrom

    Returns
    ------
    wav_vac: array
        Vacuum wavelengths in units of Angstrom
    """
    s = 1e4 / wav_air
    s2 = s ** 2
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s2) \
            + 0.0001599740894897 / (38.92568793293 - s2)
    return n * wav_air


def vac2air(wav_vac):
    """
    Vacuum to air wavelength conversion according to
    Donald Morton (2000, ApJ.Suppl., 130, 403).
    See https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    NOTE: This is valid only for wavelengths greater than 2000AA!

    Parmeters
    --------
    wav_vac: array
        Vacuum wavelengths in units of Angstrom

    Returns
    ------
    wav_air: array
        Air wavelengths in units of Angstrom
    """
    s = 1e4 / wav_vac
    s2 = s ** 2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + \
            0.00015998 / (38.9 - s2)
    return wav_vac / n
