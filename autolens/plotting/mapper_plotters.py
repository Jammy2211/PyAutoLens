import numpy as np
import matplotlib.pyplot as plt
import itertools

from autolens.plotting import tools
from autolens.inversion import mappers
from autolens.plotting import imaging_plotters
from autolens.plotting import tools_array

def plot_image_and_mapper(image, mapper, mask=None, positions=None, plot_centres=False, plot_grid=True,
                          source_pixels=None,
                          units='arcsec', kpc_per_arcsec=None,
                          output_path=None, output_filename='images', output_format='show'):

    rows, columns, figsize = tools.get_subplot_rows_columns_figsize(number_subplots=2)
    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    imaging_plotters.plot_image(image=image, mask=mask, positions=positions, grid=None, as_subplot=True,
                                units=units, kpc_per_arcsec=None, xyticksize=16,
                                norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                                figsize=None, aspect='auto', cmap='jet', cb_ticksize=10,
                                titlesize=10, xlabelsize=10, ylabelsize=10,
                                output_path=output_path, output_format=output_format)

    plt.subplot(rows, columns, 2)

    plot_mapper(mapper=mapper, plot_centres=plot_centres, plot_grid=plot_grid, source_pixels=source_pixels,
                as_subplot=True,
                units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=None)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
    plt.close()

def plot_mapper(mapper, plot_centres, plot_grid, source_pixels, as_subplot, units, kpc_per_arcsec, figsize):

    if isinstance(mapper, mappers.RectangularMapper):
        plot_rectangular_mapper(mapper=mapper, plot_centres=plot_centres, plot_grid=plot_grid,
                                source_pixels=source_pixels, as_subplot=as_subplot,
                                units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize)

def plot_rectangular_mapper(mapper, plot_centres=False, plot_grid=True, source_pixels=None, as_subplot=False,
                            units='arcsec', kpc_per_arcsec=None, figsize=(20, 15)):

    tools_array.setup_figure(figsize=figsize, as_subplot=as_subplot)

    ys = np.linspace(0, mapper.shape[0], mapper.shape[0]+1)
    xs = np.linspace(0, mapper.shape[1], mapper.shape[1]+1)

    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color='black', linestyle='-')
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color='black', linestyle='-')

    tools_array.set_xy_labels_and_ticks_in_pixels(shape=mapper.shape, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                                  xticks=mapper.geometry.xticks, yticks=mapper.geometry.yticks,
                                                  xlabelsize=16, ylabelsize=16, xyticksize=16)

    plt.ylim(0, mapper.shape[0])
    plt.xlim(0, mapper.shape[1])

    if plot_centres:
        pixel_centres = mapper.geometry.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=mapper.geometry.pixel_centres)
        plt.scatter(y=pixel_centres[:,0], x=pixel_centres[:,1], s=3, c='r')

    if plot_grid:
        grid = mapper.geometry.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=mapper.grids.image)
        plt.scatter(y=grid[:,0], x=grid[:,1], s=1)

    if source_pixels is not None:
        grid = mapper.geometry.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=mapper.grids.image)
        point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
        for source_pixel in source_pixels:
            plt.scatter(y=grid[mapper.pixelization_to_image[source_pixel],0],
                        x=grid[mapper.pixelization_to_image[source_pixel], 1], s=8,  color=next(point_colors))

    plt.show()