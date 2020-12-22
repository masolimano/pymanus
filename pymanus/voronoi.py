import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import Voronoi

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    NOTE: this function was copied from somewhere, I don't remember where
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def plot_polygons(polygons, imgdata, nodes, max_radius, xlim=(-1,0), ylim=(-24, -22), ax=None, alpha=0.5, linewidth=0.7, vmin=None, vmax=None, saveas=None, show=True, cmap='viridis'):
    """
    Plot a tesselation by ploting contiguous polygons and color them according
    to some `imgdata`

    Parameters
    ---------
    polygons: list
        List of lists of vertices of each polygon
    imgdata: 1d array
        Has to have a size equal to len(polygons)
    max_radius: float
        Polygons that have at least one vertex outside this radius
        will not be included in the plot
    show: bool
        Whether to call plt.show() or not
    saveas: str
        Filename of the saved figure. If None, the figure is
        not saved to disk.
    The rest are kwargs for the plotting routine
    """
    # Configure plot
    if ax is None:
        plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
    # Remove ticks
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    colormap = mpl.cm.get_cmap(cmap)
    if vmax == None and vmin == None:
        norm = mpl.colors.Normalize(vmin=imgdata.min(), vmax=imgdata.max())
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    for i, poly in enumerate(polygons):
        center = nodes[i]
        distances = np.sqrt(np.sum((poly - center) ** 2, axis=1))
        if np.any(distances > max_radius):
            continue
        colored_cell = mpl.patches.Polygon(poly,
                               linewidth=linewidth,
                               alpha=alpha,
                               facecolor=colormap(norm(imgdata[i])),
                               edgecolor="black")
        ax.add_patch(colored_cell)
    if not saveas is None:
        plt.savefig(saveas)
    if show:
        plt.show()
    return ax
