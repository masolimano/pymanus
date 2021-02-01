"""
--------------------------Original license-------------------------------------
The MIT License (MIT)

Copyright (c) 2015 Alex Hagen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import re
import astropy.units as u
from astropy.table import Table
from astropy.visualization import quantity_support
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(
    H0 = 70 * u.km / u.s / u.Mpc,
    Om0 = 0.3,
    Tcmb0 = 2.725 * u.K
)


class MagphysOutput(object):
    """
    Class to read and display the output of MAGPHYS into Python.
    """

    def __init__(self, fitfilename, sedfilename, obsfilename='observations.dat', fltfilename='filters.dat'):
        """
        Parameters
        ----------
        fitfilename: str
            Path of the .fit file (contains the fit results)

        sedfilename: str
            Path of the .sed file (contains the SED info and posteriors)

        obsfilename: str
            Path of the observations.dat (contains the input fluxes)
        fltfilename: str
            Path of the filters.dat file (specifies lambda_eff for each filter)
        """
        obj_name = fitfilename.removesuffix('.fit')
        fitfile = open(fitfilename)
        sedfile = open(sedfilename)
        fitinfo = fitfile.readlines()
        sedinfo = sedfile.readlines()
        fitfile.close()
        sedfile.close()

        #strip out newline characters
        for i in range(len(fitinfo)):
            fitinfo[i] = fitinfo[i].strip()
        for i in range(len(sedinfo)):
            sedinfo[i] = sedinfo[i].strip()

        # Load observed fluxes
        obs = Table.read(obsfilename, format='ascii.commented_header')
        filters_tab = Table.read(fltfilename, format='ascii.commented_header')
        row, = np.where(obs['ID'] == obj_name)[0]

        # First go through fitinfo
        filternames = fitinfo[1].strip("#")
        filternames = filternames.split()
        #flux = np.array(fitinfo[2].split(), dtype=float) * u.Jy
        #fluxerr = np.array(fitinfo[3].split(), dtype=float) * u.Jy
        flux = np.array([obs[row][flt] for flt in filternames]) * u.Jy
        fluxerr = np.array([obs[row][flt + '_err'] for flt in filternames]) * u.Jy
        predicted = np.array(fitinfo[12].split(), dtype=float) * u.Jy
        self.obs_filters = filternames
        self.obs_filters_waves = dict(zip(filters_tab['name'], filters_tab['lambda_eff'] * u.um))
        self.obs_flux = flux
        self.obs_flux_err = fluxerr
        self.obs_predict = predicted

        bestfitmodel = fitinfo[8].split()
        self.bestfit_i_sfh = int(bestfitmodel[0])
        self.bestfit_i_ir = int(bestfitmodel[1])
        self.bestfit_chi2 = float(bestfitmodel[2])
        self.bestfit_redshift = float(bestfitmodel[3])

        bestfitparams = fitinfo[9].strip('.#')
        bestfitparams = re.split('\.+', bestfitparams)
        bestfitresults = list(map(float, fitinfo[10].split()))
        assert len(bestfitparams) == len(bestfitresults)
        for i,paramname in enumerate(bestfitparams):
            setattr(self, self.clean_param_names(paramname), bestfitresults[i])

        #now working on the marginal PDF histograms for each parameter
        marginalpdfs = fitinfo[15:]
        #first, need to split the pdfs into each parameter
        self.marginal_pdfs = {}
        self.marginal_percentiles = {}
        hash_idx = []
        for i in range(len(marginalpdfs)):
            if '#' in marginalpdfs[i]:
                hash_idx.append(i)
        assert len(hash_idx) % 2 == 0
        for i in range(len(hash_idx) // 2):
            param = marginalpdfs[hash_idx[2 * i]].strip(' #.')
            marginal = marginalpdfs[hash_idx[2 * i] + 1:hash_idx[2 * i + 1]]
            marginal = np.array([j.split() for j in marginal], dtype=float)
            percentile = np.array(marginalpdfs[hash_idx[2 * i + 1] + 1].split(), dtype=float)
            self.marginal_pdfs[self.clean_param_names(param)] = marginal
            self.marginal_percentiles[self.clean_param_names(param)] = percentile

        #now time for the SED file
        self.sed_model_params = {}
        #there are model names and params on lines 2 & 3 and 5 & 6
        modelparams = sedinfo[2].strip('.#')
        modelparams = re.split('\.+', modelparams)
        model_vals = list(map(float,sedinfo[3].split()))
        assert len(modelparams) == len(model_vals)
        for i,paramname in enumerate(modelparams):
            self.sed_model_params[self.clean_param_names(paramname)] = model_vals[i]
        modelparams = sedinfo[5].strip('.#')
        modelparams = re.split('\.+', modelparams)
        model_vals = list(map(float,sedinfo[6].split()))
        assert len(modelparams) == len(model_vals)
        for i,paramname in enumerate(modelparams):
            self.sed_model_params[self.clean_param_names(paramname)] = model_vals[i]
        #sed is from line 10 to the end. 
        #three columns, log lambda, log L_lam attenuated, log L_lam unattenuated
        model_sed = sedinfo[10:]
        model_sed = [i.split() for i in model_sed]
        self.sed_model = np.array(model_sed,dtype=float)
        self.sed_model_logwaves = self.sed_model[:, 0] * u.dex(u.AA)
        self.sed_model_logluminosity_lambda = self.sed_model[:, 1] * u.dex(u.Lsun / u.AA)
        self.sed_model_logluminosity_lambda_unatt = self.sed_model[:, 2] * u.dex(u.Lsun / u.AA)

    @staticmethod
    def clean_param_names(paramname):
        """
        this removes the character '()/^*' from param names
        """
        paramname = paramname.replace('(', '')
        paramname = paramname.replace(')', '')
        paramname = paramname.replace('/', '_')
        paramname = paramname.replace('^', '_')
        paramname = paramname.replace('*', 'star')
        return paramname

    def plot(self, ax=None, lmin=None, lmax=None, lunit=u.um, unatt=False, **kwargs):
        """
        Method for quick visualization of the best fit SED model.

        Parameters
        ----------
        ax: Axes instance, optional
           Previously instantiated Axes object to draw the artists to.
           Defaults `None`, which grabs the current axes or creates a new one.

        lmin, lmax: floats, optional
            Limits of the wavelength range to display. Assumes `lunit` as the wavelength unit.

        lunit: astropy.Unit instance, optional
            Wavelength unit, defaults to micrometers.

        unatt: bool, optional
            Whether to plot the unattenuated model stellar emission.

        kwargs: dict, optional
            Keyword arguments passed to `ax.plot`.
        """

        # Create an Axes instance for the plot?
        if ax == None:
            ax = plt.gca()

        # The output SED of Magphys is given in units of dex(Lsol/AA)
        # here we convert it to observed flux density in Jy
        wave = self.sed_model_logwaves.physical.to(lunit)
        model_fnu = self.sed_model_logluminosity_lambda.physical
        #model_fnu *= 1 / (1 + self.bestfit_redshift)
        model_fnu *= 1 / (4 * np.pi * cosmo.luminosity_distance(self.bestfit_redshift) ** 2)
        model_fnu = model_fnu.to(u.Jy, equivalencies=u.spectral_density(wave))
        wherevalid = np.where(model_fnu > 1e-8 * u.Jy)
        wave = wave[wherevalid]
        model_fnu = model_fnu[wherevalid]

        # Crop to selected wavelength range
        if lmin is not None or lmax is not None:
            ww = np.where(
                np.logical_and(
                    wave > lmin * lunit,
                    wave < lmax * lunit
                )
            )
            wave = wave[ww]
            model_fnu = model_fnu[ww]

        if unatt:
            model_fnu_unatt = self.sed_model_logluminosity_lambda_unatt.physical[wherevalid][ww]
            model_fnu_unatt *= 1 / (4 * np.pi * cosmo.luminosity_distance(self.bestfit_redshift) ** 2)
            model_fnu_unatt = model_fnu_unatt.to(u.Jy, equivalencies=u.spectral_density(wave))

        # Default style dictionaries for errorbars
        # TODO: make the user able to pass her own kwargs
        errkw = dict(linestyle='', color='purple', capsize=6, elinewidth=2, marker='o', mec='purple', mfc='white', mew=2)
        uplimkw = dict(linestyle='', color='purple', capsize=5, marker='_',  ms=14, mew=2, lw=2)

        # The actual plot
        with quantity_support():
            for i, flt in enumerate(self.obs_filters):
                fnu = self.obs_flux[i]
                fnu_err = self.obs_flux_err[i]
                lbda_eff = self.obs_filters_waves[flt].to(lunit)

                if fnu > 0:
                    ax.errorbar([lbda_eff.value], [fnu.value], yerr=[fnu_err.value], **errkw)

                else:
                    ax.errorbar([lbda_eff.value], [fnu_err.value], yerr=[0.2 * fnu_err.value], uplims=True, **uplimkw)

            ax.errorbar([], [], yerr=[], label='Observed flux', **errkw)
            ax.plot(wave, model_fnu, label='Best fit SED', **kwargs)
            ylim = ax.get_ylim()
            if unatt:
                # Color and zorder of the unattenuated stellar SED
                # are hard-coded
                kwargs['color'] = 'lightgray'
                kwargs['zorder'] = -1
                ax.plot(wave, model_fnu_unatt, label='Unatt. starlight', **kwargs)
            ax.set_ylim(model_fnu.min(), ylim[1])
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()




if __name__ == '__main__':
    import os
    plt.style.use('ticky')
    os.chdir('/home/manuel/arcos/SGASJ1226/sedfit/magphys_run14/')
    result = MagphysOutput('ARC_NORTH.fit', 'ARC_NORTH.sed')
    fig, ax = plt.subplots(figsize=(7, 4))
    result.plot(ax=ax, lmin=0.4, lmax=3000, unatt=True, color='tab:orange', zorder=0)
    ax.set_ylim(1e-5, 4e-3)
    ax.set_xlim(0.4, 2500)
    plt.show()
