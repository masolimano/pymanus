import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import regions

def extract(cube_filename, region_filename):
    """
    cube_filename has to be a multi-extension FITS file whose
    second extension (1th index) contains the cube data
    """
    with fits.open(cube_filename) as cube_hdul:
        cube_hdr = cube_hdul[1].header
        assert cube_hdr['NAXIS'] == 3 # Checks whether this is actually a cube or not
        celestial_wcs = WCS(cube_hdr).celestial
        spectral_wcs = WCS(cube_hdr).spectral
        regs_sky = regions.read_ds9(region_filename)

        for reg in regs_sky:
            center = reg.center.to_string('hmsdms').replace(' ', '')
            reg_pix = reg.to_pixel(celestial_wcs)
            mask = reg_pix.to_mask().to_image(shape=cube_hdul[1].data.shape[1:])
            w = np.where(mask)
            spec = np.sum(cube_hdul[1].data[:, w[0], w[1]], axis=1)
            varspec = np.sum(cube_hdul[2].data[:, w[0], w[1]], axis=1)

            # This approach might lose the actual
            # unit for the spectral axis. TODO: test
            spec_hdr = spectral_wcs.to_header()
            spec_hdr['OBJECT'] = cube_hdr['OBJECT']
            spec_hdr['EXTNAME'] = cube_hdr['DATA']
            spec_hdr['BUNIT'] = cube_hdr['BUNIT']

            var_hdr = spectral_wcs.to_header()
            var_hdr['OBJECT'] = cube_hdr['OBJECT']
            var_hdr['EXTNAME'] = cube_hdr['STAT']
            var_hdr['BUNIT'] = cube_hdul[2].header['BUNIT']

            primary = fits.PrimaryHDU(spec, header=spec_hdr)
            stat = fits.ImageHDU(varspec, header=var_hdr)
            return fits.HDUList([primary, stat]) # BUG: <-- this will only yield the last spectrum

def main(args=None):
    pass


