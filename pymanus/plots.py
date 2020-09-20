import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

class AnchoredBeam(mpl.offsetbox.AnchoredOffsetbox):
    def __init__(self, transform, bmin, bmaj, angle, loc,
                 pad=0.1, borderpad=0.1, prop=None, frameon=False, **kwargs):
        """
        Returns a beam patch defined in data coordinates anchored to the given axes.
        Also draws the major and minor axes of the ellipse.

        Adapted from mpl_toolkts.axes_grid1.anchored_artists.AnchoredEllipse
        pad, borderpad in fraction of the legend font size (or prop)

        Parameters
        ---------
        transform: Transform
            Use ax.transData to use data coordinates

        bmin: float
            Beam minor axis length in pixel units

        bmaj: float
            Beam minor axis length in pixel units

        angle: float
            Beam position angle in degrees East of North

        loc: str or int
            Location of anchored artist

        **kwargs:
            Passed to ~matplotlib.patches.Ellipse.


        """

        angle = -angle # make counter-clockwise
        self._box = mpl.offsetbox.AuxTransformBox(transform)

        self.ellipse = mpl.patches.Ellipse((0, 0), bmin, bmaj, angle, fill=False, **kwargs)

        semiminor = bmin / 2
        semimajor = bmaj / 2
        cos_a = np.cos(angle * np.pi / 180 + np.pi / 2)
        sin_a = np.sin(angle * np.pi / 180 + np.pi / 2)


        self.ydata_maj = [- semimajor * sin_a,    semimajor * sin_a]
        self.xdata_maj = [- semimajor * cos_a,    semimajor * cos_a]
        self.ydata_min = [- semiminor * cos_a,    semiminor * cos_a]
        self.xdata_min = [  semiminor * sin_a,  - semiminor * sin_a]

        # Create Line2D artists (cross)
        cross_kwargs = dict(linewidth=self.ellipse._linewidth,
                            color=self.ellipse._edgecolor,
                            solid_capstyle='butt')

        majoraxis = mpl.lines.Line2D(ydata=self.ydata_maj, xdata=self.xdata_maj, **cross_kwargs)
        minoraxis = mpl.lines.Line2D(ydata=self.ydata_min, xdata=self.xdata_min, **cross_kwargs)



        self._box.add_artist(self.ellipse)
        self._box.add_artist(majoraxis)
        self._box.add_artist(minoraxis)

        super().__init__(loc, pad=pad, borderpad=borderpad,
                         child=self._box, prop=prop, frameon=frameon)
