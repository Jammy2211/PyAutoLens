from matplotlib import pyplot as plt

from autolens import conf
from autolens import exc
from autolens.plotting import plotters

def plot_plane_grid(plane, xmin=None, xmax=None, ymin=None, ymax=None):

    plt.figure(figsize=(12, 8))
    plt.scatter(y=plane.grids.image[:, 0], x=plane.grids.image[:, 1], marker='.', s=1)
    plotters.set_title(title='Plane Grid', titlesize=36)
    plt.xlabel('x (arcsec)', fontsize=20)
    plt.ylabel('y (arcsec)', fontsize=20)
    plt.tick_params(labelsize=20)
    if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
        plt.axis([xmin, xmax, ymin, ymax])
    plt.show()
    plt.close()