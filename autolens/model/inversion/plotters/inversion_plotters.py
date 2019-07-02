import autofit as af
import matplotlib
backend = af.conf.instance.visualize.get('figures', 'backend', str)
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autolens.plotters import plotter_util, array_plotters
from autolens.model.inversion.plotters import mapper_plotters
from autolens.model.inversion import mappers

def plot_inversion_subplot(
        inversion, mask=None, positions=None, grid=None,  extract_array_from_mask=False, zoom_around_mask=False,
        units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        output_path=None, output_format='show', output_filename='inversion_subplot'):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=6)

    if figsize is None:
        figsize = figsize_tool

    ratio = \
        float((inversion.mapper.geometry.arc_second_maxima[1] -
               inversion.mapper.geometry.arc_second_minima[1]) / \
              (inversion.mapper.geometry.arc_second_maxima[0] -
               inversion.mapper.geometry.arc_second_minima[0]))

    if aspect is 'square':
        aspect_inv = ratio
    elif aspect is 'auto':
        aspect_inv = 1.0 / ratio
    elif aspect is 'equal':
        aspect_inv = 1.0

    plt.figure(figsize=figsize)

    plt.subplot(rows, columns, 1)

    plot_reconstructed_image(
        inversion=inversion, mask=mask, positions=positions, grid=grid, as_subplot=True,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plt.subplot(rows, columns, 2, aspect=float(aspect_inv))

    plot_pixelization_values(
        inversion=inversion, positions=None, should_plot_grid=False, should_plot_centres=False, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=None,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(rows, columns, 3, aspect=float(aspect_inv))

    plot_pixelization_residual_map(
        inversion=inversion, positions=None, should_plot_grid=False, should_plot_centres=False, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=None,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(rows, columns, 4, aspect=float(aspect_inv))

    plot_pixelization_normalized_residual_map(
        inversion=inversion, positions=None, should_plot_grid=False, should_plot_centres=False, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=None,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(rows, columns, 5, aspect=float(aspect_inv))

    plot_pixelization_chi_squared_map(
        inversion=inversion, positions=None, should_plot_grid=False, should_plot_centres=False, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=None,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(rows, columns, 6, aspect=float(aspect_inv))

    plot_pixelization_regularization_weights(
        inversion=inversion, positions=None, should_plot_grid=False, should_plot_centres=False, as_subplot=True,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=None,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_filename=None, output_format=output_format)

    plotter_util.output_subplot_array(
        output_path=output_path, output_filename=output_filename, output_format=output_format)

    plt.close()

def plot_reconstructed_image(
        inversion, mask=None, positions=None, grid=None, as_subplot=False,
        extract_array_from_mask=False, zoom_around_mask=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Reconstructed Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='reconstructed_inversion_image'):

    array_plotters.plot_array(
        array=inversion.reconstructed_data_2d, mask=mask, positions=positions, grid=grid, as_subplot=as_subplot,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
        linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_pixelization_values(
        inversion, plot_origin=True, positions=None, should_plot_centres=False,
        should_plot_grid=False, should_plot_border=False, image_pixels=None,
        source_pixels=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
        linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Reconstructed Pixelization', titlesize=16, xlabelsize=16, ylabelsize=16,
        xyticksize=16,
        output_path=None, output_format='show', output_filename='inversion_pixelization_values'):

    if output_format is 'fits':
        return

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    plot_inversion_with_source_values(
        inversion=inversion, source_pixel_values=inversion.pixelization_values,
        plot_origin=plot_origin, positions=positions, should_plot_centres=should_plot_centres,
        should_plot_grid=should_plot_grid, should_plot_border=should_plot_border,
        image_pixels=image_pixels, source_pixels=source_pixels, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plotter_util.close_figure(as_subplot=as_subplot)

def plot_pixelization_residual_map(
        inversion, plot_origin=True, positions=None, should_plot_centres=False,
        should_plot_grid=False, should_plot_border=False, image_pixels=None,
        source_pixels=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
        linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Reconstructed Pixelization Residual-Map', titlesize=16, xlabelsize=16, ylabelsize=16,
        xyticksize=16,
        output_path=None, output_format='show', output_filename='inversion_pixelization_residual_map'):

    if output_format is 'fits':
        return

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    plot_inversion_with_source_values(
        inversion=inversion, source_pixel_values=inversion.pixelization_residual_map,
        plot_origin=plot_origin, positions=positions, should_plot_centres=should_plot_centres,
        should_plot_grid=should_plot_grid, should_plot_border=should_plot_border,
        image_pixels=image_pixels, source_pixels=source_pixels, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plotter_util.close_figure(as_subplot=as_subplot)

def plot_pixelization_normalized_residual_map(
        inversion, plot_origin=True, positions=None, should_plot_centres=False,
        should_plot_grid=False, should_plot_border=False, image_pixels=None,
        source_pixels=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
        linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Reconstructed Pixelization Normalized Residual Map', titlesize=16, xlabelsize=16, ylabelsize=16,
        xyticksize=16,
        output_path=None, output_format='show', output_filename='inversion_pixelization_normalized_residual_map'):

    if output_format is 'fits':
        return

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    plot_inversion_with_source_values(
        inversion=inversion, source_pixel_values=inversion.pixelization_normalized_residual_map,
        plot_origin=plot_origin, positions=positions, should_plot_centres=should_plot_centres,
        should_plot_grid=should_plot_grid, should_plot_border=should_plot_border,
        image_pixels=image_pixels, source_pixels=source_pixels, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plotter_util.close_figure(as_subplot=as_subplot)

def plot_pixelization_chi_squared_map(
        inversion, plot_origin=True, positions=None, should_plot_centres=False,
        should_plot_grid=False, should_plot_border=False, image_pixels=None,
        source_pixels=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
        linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Reconstructed Pixelization Chi-Squared Map', titlesize=16, xlabelsize=16, ylabelsize=16,
        xyticksize=16,
        output_path=None, output_format='show', output_filename='inversion_pixelization_chi_squared_map'):

    if output_format is 'fits':
        return

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    plot_inversion_with_source_values(
        inversion=inversion, source_pixel_values=inversion.pixelization_chi_squared_map,
        plot_origin=plot_origin, positions=positions, should_plot_centres=should_plot_centres,
        should_plot_grid=should_plot_grid, should_plot_border=should_plot_border,
        image_pixels=image_pixels, source_pixels=source_pixels, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plotter_util.close_figure(as_subplot=as_subplot)


def plot_pixelization_regularization_weights(
        inversion, plot_origin=True, positions=None, should_plot_centres=False,
        should_plot_grid=False, should_plot_border=False, image_pixels=None,
        source_pixels=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
        linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Reconstructed Pixelization Regularization Weights', titlesize=16, xlabelsize=16, ylabelsize=16,
        xyticksize=16,
        output_path=None, output_format='show', output_filename='inversion_pixelization_regularization_weights'):

    if output_format is 'fits':
        return

    plotter_util.setup_figure(
        figsize=figsize, as_subplot=as_subplot)

    regularization_weights = inversion.regularization.regularization_weights_from_mapper(
        mapper=inversion.mapper)

    plot_inversion_with_source_values(
        inversion=inversion, source_pixel_values=regularization_weights,
        plot_origin=plot_origin, positions=positions, should_plot_centres=should_plot_centres,
        should_plot_grid=should_plot_grid, should_plot_border=should_plot_border,
        image_pixels=image_pixels, source_pixels=source_pixels, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

    plotter_util.close_figure(as_subplot=as_subplot)

def plot_inversion_with_source_values(
        inversion, source_pixel_values,
        plot_origin=True, positions=None, should_plot_centres=False,
        should_plot_grid=False, should_plot_border=False, image_pixels=None,
        source_pixels=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
        linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Reconstructed Pixelization', titlesize=16, xlabelsize=16, ylabelsize=16,
        xyticksize=16,
        output_path=None, output_format='show', output_filename='pixelization_source_values'):

    if isinstance(inversion.mapper, mappers.RectangularMapper):

        reconstructed_pixelization = \
            inversion.mapper.reconstructed_pixelization_from_solution_vector(solution_vector=source_pixel_values)

        origin = get_origin(image=reconstructed_pixelization, plot_origin=plot_origin)

        array_plotters.plot_array(
            array=reconstructed_pixelization, origin=origin, positions=positions, as_subplot=True,
            units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
            linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
            cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
            title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            output_filename=output_filename)

        mapper_plotters.plot_rectangular_mapper(
            mapper=inversion.mapper,
            should_plot_centres=should_plot_centres, should_plot_grid=should_plot_grid,
            should_plot_border=should_plot_border,
            image_pixels=image_pixels, source_pixels=source_pixels,
            as_subplot=True,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            title=title, titlesize=titlesize, xlabelsize=xlabelsize,
            ylabelsize=ylabelsize, xyticksize=xyticksize)

        plotter_util.output_figure(
            array=reconstructed_pixelization, as_subplot=as_subplot,
            output_path=output_path, output_filename=output_filename, output_format=output_format)

    elif isinstance(inversion.mapper, mappers.VoronoiMapper):

        mapper_plotters.plot_voronoi_mapper(
            mapper=inversion.mapper, source_pixel_values=source_pixel_values,
            should_plot_centres=should_plot_centres,
            should_plot_grid=should_plot_grid, should_plot_border=should_plot_border,
            image_pixels=image_pixels, source_pixels=source_pixels,
            as_subplot=True,
            units=units, kpc_per_arcsec=kpc_per_arcsec,
            title=title, titlesize=titlesize, xlabelsize=xlabelsize,
            ylabelsize=ylabelsize, xyticksize=xyticksize)

        plotter_util.output_figure(
            array=None, as_subplot=as_subplot,
            output_path=output_path, output_filename=output_filename, output_format=output_format)

def get_origin(image, plot_origin):

    if plot_origin:
        return image.origin
    else:
        return None