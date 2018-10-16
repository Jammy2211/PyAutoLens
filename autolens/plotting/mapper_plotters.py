import numpy as np
import matplotlib.pyplot as plt
import itertools

from autolens.plotting import tools
from autolens.inversion import mappers
from autolens.plotting import plot_grid
from autolens.plotting import imaging_plotters
from autolens.plotting import tools_array

def plot_image_and_mapper(image, mapper, mask=None, positions=None, should_plot_centres=False, should_plot_grid=True,
                          image_pixels=None, source_pixels=None,
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

    point_colors = itertools.cycle(["w", "c", "y", "r", "k", "b", "g", "m"])
    for source_pixel_set in source_pixels:
        color = next(point_colors)
        for source_pixel in source_pixel_set:
            image_pixel = mapper.pixelization_to_image[source_pixel]
            image_pixel_2d = mapper.grids.image.mask.grid_to_pixel[image_pixel]
            plt.scatter(y=image_pixel_2d[0, 0], x=image_pixel_2d[0, 1], color=color, s=10.0)

    plt.subplot(rows, columns, 2)

    plot_mapper(mapper=mapper, should_plot_centres=should_plot_centres, should_plot_grid=should_plot_grid,
                image_pixels=image_pixels, source_pixels=source_pixels,
                as_subplot=True,
                units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=None)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
    plt.close()

def plot_mapper(mapper, should_plot_centres, should_plot_grid, image_pixels, source_pixels, as_subplot,
                units, kpc_per_arcsec, figsize):

    if isinstance(mapper, mappers.RectangularMapper):
        plot_rectangular_mapper(mapper=mapper, should_plot_centres=should_plot_centres,
                                should_plot_grid=should_plot_grid,
                                image_pixels=image_pixels, source_pixels=source_pixels, as_subplot=as_subplot,
                                units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize)

def plot_rectangular_mapper(mapper, should_plot_centres=False, should_plot_grid=True,
                            image_pixels=None, source_pixels=None, as_subplot=False,
                            units='arcsec', kpc_per_arcsec=None, figsize=(20, 15)):

    tools.setup_figure(figsize=figsize, as_subplot=as_subplot)

    plot_pixelization_lines(mapper=mapper)

    plot_grid.set_xy_labels_and_ticks_in_arcsec(units=units, kpc_per_arcsec=kpc_per_arcsec,
                                                xticks=mapper.geometry.xticks, yticks=mapper.geometry.yticks,
                                                xlabelsize=16, ylabelsize=16, xyticksize=16)

    set_limits(mapper=mapper)

    if should_plot_centres:
        pixel_centres = mapper.geometry.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=mapper.geometry.pixel_centres)
        plt.scatter(y=pixel_centres[:,0], x=pixel_centres[:,1], s=3, c='r')

    if should_plot_grid:
        plot_grid.plot_grid(grid=mapper.grids.image, as_subplot=as_subplot,
                            pointsize=5, xyticksize=10,
                            title='Source-Plane Grid', titlesize=10, xlabelsize=10, ylabelsize=10)

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    plot_image_pixels(mapper=mapper, image_pixels=image_pixels, point_colors=point_colors)
    plot_source_pixels(mapper=mapper, source_pixels=source_pixels, point_colors=point_colors)

    plt.show()

def plot_pixelization_lines(mapper):

    ys = np.linspace(mapper.geometry.arc_second_minima[0], mapper.geometry.arc_second_maxima[0], mapper.shape[0]+1)
    xs = np.linspace(mapper.geometry.arc_second_minima[1], mapper.geometry.arc_second_maxima[1], mapper.shape[1]+1)

    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color='black', linestyle='-')
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color='black', linestyle='-')

def set_limits(mapper):

    plt.ylim(mapper.geometry.arc_second_minima[0], mapper.geometry.arc_second_maxima[0])
    plt.xlim(mapper.geometry.arc_second_minima[0], mapper.geometry.arc_second_maxima[0])

def plot_image_pixels(mapper, image_pixels, point_colors):

    if image_pixels is not None:
        for image_pixel_set in image_pixels:
            color = next(point_colors)
            plt.scatter(y=mapper.grids.image[[image_pixel_set],0],
                        x=mapper.grids.image[[image_pixel_set],1], s=8, color=color)

def plot_source_pixels(mapper, source_pixels, point_colors):

    if source_pixels is not None:
        for source_pixel_set in source_pixels:
            color = next(point_colors)
            for source_pixel in source_pixel_set:
                plt.scatter(y=mapper.grids.image[mapper.pixelization_to_image[source_pixel],0],
                            x=mapper.grids.image[mapper.pixelization_to_image[source_pixel],1], s=8,
                            color=color)