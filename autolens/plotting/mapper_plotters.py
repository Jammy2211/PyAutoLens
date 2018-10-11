import numpy as np
import matplotlib.pyplot as plt
from autolens.plotting import array_plotters

def plot_rectangular_mapper(mapper, plot_grid=False, units='arcsec', kpc_per_arcsec=None):

    plt.figure(figsize=(20, 15))

    ys = np.linspace(0, mapper.shape[0], mapper.shape[0])
    xs = np.linspace(0, mapper.shape[1], mapper.shape[1])

    plt.xlim(0.0, 15.0)

    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color='black', linestyle='-')
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color='black', linestyle='-')

    array_plotters.set_xy_labels_and_ticks(shape=mapper.shape, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                           xticks=mapper.geometry.xticks, yticks=mapper.geometry.yticks,
                                           xlabelsize=16, ylabelsize=16, xyticksize=16)

    if plot_grid:
        print(mapper.grids.image)
    #    grid = mapper.geometry.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=mapper.grids.image)
     #   print(grid)
        plt.scatter(y=mapper.grids.image[:, 0], x=mapper.grids.image[:, 1], s=1)

    plt.show()