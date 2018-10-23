import numpy as np
import matplotlib.pyplot as plt
import itertools

from autolens.plotting import tools
from autolens.inversion import mappers
from autolens.plotting import tools_grid
from autolens.plotting import tools_array
from autolens.plotting import imaging_plotters

def plot_image_and_mapper(image, mapper, mask=None, positions=None, should_plot_centres=False, should_plot_grid=True,
                          image_pixels=None, source_pixels=None,
                          units='arcsec', kpc_per_arcsec=None,
                          output_path=None, output_filename='image_and_mapper', output_format='show'):

    rows, columns, figsize = tools.get_subplot_rows_columns_figsize(number_subplots=2)
    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    imaging_plotters.plot_image(image=image, mask=mask, positions=positions, grid=None, as_subplot=True,
                                units=units, kpc_per_arcsec=None, xyticksize=16,
                                norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                                figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
                                titlesize=10, xlabelsize=10, ylabelsize=10,
                                output_path=output_path, output_format=output_format)

    image_grid = convert_grid(grid=mapper.grids.image.unlensed_grid, units=units, kpc_per_arcsec=kpc_per_arcsec)

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    plot_image_plane_image_pixels(grid=image_grid, image_pixels=image_pixels, point_colors=point_colors)
    plot_image_plane_source_pixels(grid=image_grid, mapper=mapper, source_pixels=source_pixels,
                                   point_colors=point_colors)

    plt.subplot(rows, columns, 2)

    plot_mapper(mapper=mapper, should_plot_centres=should_plot_centres, should_plot_grid=should_plot_grid,
                image_pixels=image_pixels, source_pixels=source_pixels,
                as_subplot=True,
                units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=None)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
    plt.close()

def plot_mapper(mapper, should_plot_centres=False, should_plot_grid=True,
                image_pixels=None, source_pixels=None, as_subplot=False,
                units='arcsec', kpc_per_arcsec=None,
                xyticksize=16, figsize=(7, 7),
                title='Mapper', titlesize=16, xlabelsize=16, ylabelsize=16,
                output_path=None, output_filename='mapper', output_format='show'):

    if isinstance(mapper, mappers.RectangularMapper):

        plot_rectangular_mapper(mapper=mapper, should_plot_centres=should_plot_centres,
                                should_plot_grid=should_plot_grid,
                                image_pixels=image_pixels, source_pixels=source_pixels, as_subplot=as_subplot,
                                units=units, kpc_per_arcsec=kpc_per_arcsec,
                                xyticksize=xyticksize, figsize=figsize,
                                title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_rectangular_mapper(mapper, should_plot_centres=False, should_plot_grid=True,
                            image_pixels=None, source_pixels=None, as_subplot=False,
                            units='arcsec', kpc_per_arcsec=None,
                            xyticksize=16, figsize=(7, 7),
                            title='Rectangular Mapper', titlesize=16, xlabelsize=16, ylabelsize=16,
                            output_path=None, output_filename='rectangular_mapper', output_format='show'):

    tools.setup_figure(figsize=figsize, as_subplot=as_subplot)

    set_axis_limits(mapper=mapper, units=units, kpc_per_arcsec=kpc_per_arcsec)
    plot_pixelization_lines(mapper=mapper, units=units, kpc_per_arcsec=kpc_per_arcsec)

    tools.set_title(title=title, titlesize=titlesize)
    tools_grid.set_xy_labels_and_ticks_in_arcsec(units=units, kpc_per_arcsec=kpc_per_arcsec,
                                                 xticks=mapper.geometry.xticks, yticks=mapper.geometry.yticks,
                                                 xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)

    plot_centres(should_plot_centres=should_plot_centres, mapper=mapper, units=units, kpc_per_arcsec=kpc_per_arcsec)

    plot_plane_grid(should_plot_grid=should_plot_grid, mapper=mapper, as_subplot=True, units=units,
                    kpc_per_arcsec=kpc_per_arcsec, pointsize=10, xyticksize=xyticksize,
                    title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize)

    mapper_grid = convert_grid(grid=mapper.grids.image, units=units, kpc_per_arcsec=kpc_per_arcsec)

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    plot_source_plane_image_pixels(grid=mapper_grid, image_pixels=image_pixels, point_colors=point_colors)
    plot_source_plane_source_pixels(grid=mapper_grid, mapper=mapper, source_pixels=source_pixels,
                                    point_colors=point_colors)

    tools.output_figure(None, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_pixelization_lines(mapper, units, kpc_per_arcsec):

    if units is 'arcsec' or kpc_per_arcsec is None:

        ys = np.linspace(mapper.geometry.arc_second_minima[0], mapper.geometry.arc_second_maxima[0], mapper.shape[0]+1)
        xs = np.linspace(mapper.geometry.arc_second_minima[1], mapper.geometry.arc_second_maxima[1], mapper.shape[1]+1)

    elif units is 'kpc':

        ys = np.linspace(mapper.geometry.arc_second_minima[0]*kpc_per_arcsec,
                         mapper.geometry.arc_second_maxima[0]*kpc_per_arcsec, mapper.shape[0]+1)
        xs = np.linspace(mapper.geometry.arc_second_minima[1]*kpc_per_arcsec,
                         mapper.geometry.arc_second_maxima[1]*kpc_per_arcsec, mapper.shape[1]+1)

    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color='black', linestyle='-')
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color='black', linestyle='-')

def set_axis_limits(mapper, units, kpc_per_arcsec):

    if units is 'arcsec' or kpc_per_arcsec is None:

        tools_grid.set_axis_limits(axis_limits=[mapper.geometry.arc_second_minima[1],
                                                mapper.geometry.arc_second_maxima[1],
                                                mapper.geometry.arc_second_minima[0],
                                                mapper.geometry.arc_second_maxima[0]])

    elif units is 'kpc':

        tools_grid.set_axis_limits(axis_limits=[mapper.geometry.arc_second_minima[1]*kpc_per_arcsec,
                                                mapper.geometry.arc_second_maxima[1]*kpc_per_arcsec,
                                                mapper.geometry.arc_second_minima[0]*kpc_per_arcsec,
                                                mapper.geometry.arc_second_maxima[0]*kpc_per_arcsec])

def plot_centres(should_plot_centres, mapper, units, kpc_per_arcsec):

    if should_plot_centres:

        if units is 'arcsec' or kpc_per_arcsec is None:

            pixel_centres = mapper.geometry.pixel_centres

        elif units is 'kpc':

            pixel_centres = mapper.geometry.pixel_centres * kpc_per_arcsec

        plt.scatter(y=pixel_centres[:,0], x=pixel_centres[:,1], s=3, c='r')

def plot_plane_grid(should_plot_grid, mapper, as_subplot, units, kpc_per_arcsec, pointsize, xyticksize, title,
                    titlesize, xlabelsize, ylabelsize):

    if should_plot_grid:

        tools_grid.plot_grid(grid=mapper.grids.image, as_subplot=as_subplot, units=units, kpc_per_arcsec=kpc_per_arcsec,
                             pointsize=5, xyticksize=xyticksize, title=title, titlesize=titlesize,
                             xlabelsize=xlabelsize, ylabelsize=ylabelsize)

def plot_image_plane_image_pixels(grid, image_pixels, point_colors):

    if image_pixels is not None:

        for image_pixel_set in image_pixels:
            color = next(point_colors)
            plt.scatter(y=grid[image_pixel_set, 0], x=grid[image_pixel_set, 1], color=color, s=10.0)

def plot_image_plane_source_pixels(grid, mapper, source_pixels, point_colors):

    if source_pixels is not None:

        for source_pixel_set in source_pixels:
            color = next(point_colors)
            for source_pixel in source_pixel_set:
                plt.scatter(y=np.asarray(grid[mapper.pixelization_to_image[source_pixel],0]),
                            x=np.asarray(grid[mapper.pixelization_to_image[source_pixel],1]), s=8, color=color)

def plot_source_plane_image_pixels(grid, image_pixels, point_colors):

    if image_pixels is not None:

        for image_pixel_set in image_pixels:
            color = next(point_colors)
            plt.scatter(y=grid[[image_pixel_set],0], x=grid[[image_pixel_set],1], s=8, color=color)

def plot_source_plane_source_pixels(grid, mapper, source_pixels, point_colors):

    if source_pixels is not None:

        for source_pixel_set in source_pixels:
            color = next(point_colors)
            for source_pixel in source_pixel_set:
                plt.scatter(y=np.asarray(grid[mapper.pixelization_to_image[source_pixel],0]),
                            x=np.asarray(grid[mapper.pixelization_to_image[source_pixel],1]), s=8, color=color)

def convert_grid(grid, units, kpc_per_arcsec):

    if units is 'arcsec' or kpc_per_arcsec is None:
        return grid
    elif units is 'kpc':
        return grid * kpc_per_arcsec
