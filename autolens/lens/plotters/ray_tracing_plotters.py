import autofit as af
import matplotlib
backend = af.conf.instance.visualize.get('figures', 'backend', str)
matplotlib.use(backend)
from matplotlib import pyplot as plt

from autolens.plotters import plotter_util, array_plotters
from autolens.lens.plotters import plane_plotters

def plot_ray_tracing_subplot(
        tracer, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        units='arcsec', figsize=None, aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
        mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1.0,
        output_path=None, output_filename='tracer', output_format='show'):
    """Plot the observed _tracer of an analysis, using the *CCD* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    tracer : autolens.ccd.tracer.CCD
        Class containing the _tracer, noise_map-mappers and PSF that are to be plotted.
        The font size of the figure ylabel.
    output_path : str
        The path where the _tracer is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the _tracer is output. File formats (e.g. png, fits) output the _tracer to harddisk. 'show' displays the _tracer \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=6)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_image_plane_image(
        tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
        positions=positions, as_subplot=True,
        units=units, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    if tracer.has_mass_profile:

        plt.subplot(rows, columns, 2)

        plot_convergence(
            tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask, as_subplot=True,
            units=units, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            output_path=output_path, output_filename='', output_format=output_format)

        plt.subplot(rows, columns, 3)

        plot_potential(
            tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask, as_subplot=True,
            units=units, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 4)

    plane_plotters.plot_plane_image(
        plane=tracer.source_plane, as_subplot=True, positions=None, plot_grid=False,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        grid_pointsize=grid_pointsize,
        output_path=output_path, output_filename='', output_format=output_format)

    if tracer.has_mass_profile:

        plt.subplot(rows, columns, 5)

        plot_deflections_y(
            tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask, as_subplot=True,
            units=units, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            output_path=output_path, output_filename='', output_format=output_format)

        plt.subplot(rows, columns, 6)

        plot_deflections_x(
            tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask, as_subplot=True,
            units=units, figsize=figsize, aspect=aspect,
            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
            output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()

def plot_ray_tracing_individual(
        tracer, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None,
        should_plot_image_plane_image=False,
        should_plot_source_plane=False,
        should_plot_convergence=False,
        should_plot_potential=False,
        should_plot_deflections=False,
        units='arcsec',
        output_path=None, output_format='show'):
    """Plot the observed _tracer of an analysis, using the *CCD* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    tracer : autolens.ccd.tracer.CCD
        Class containing the _tracer, noise_map-mappers and PSF that are to be plotted.
        The font size of the figure ylabel.
    output_path : str
        The path where the _tracer is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the _tracer is output. File formats (e.g. png, fits) output the _tracer to harddisk. 'show' displays the _tracer \
        in the python interpreter window.
    """

    if should_plot_image_plane_image:

        plot_image_plane_image(
            tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask, positions=positions,
            units=units,
            output_path=output_path, output_format=output_format)

    if should_plot_convergence:

        plot_convergence(
            tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format=output_format)

    if should_plot_potential:

        plot_potential(
            tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format=output_format)

    if should_plot_source_plane:

        plane_plotters.plot_plane_image(
            plane=tracer.source_plane, positions=None, plot_grid=False,
            units=units,
            output_path=output_path, output_filename='tracer_source_plane', output_format=output_format)

    if should_plot_deflections:

        plot_deflections_y(
            tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format=output_format)

    if should_plot_deflections:

        plot_deflections_x(
            tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format=output_format)


def plot_image_plane_image(
        tracer, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None, as_subplot=False,
        units='arcsec', figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Tracer CCD-Plane CCD', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10, position_pointsize=10.0,
        output_path=None, output_format='show', output_filename='tracer_image_plane_image'):

    array_plotters.plot_array(
        array=tracer.profile_image_plane_image_2d, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, positions=positions,
        as_subplot=as_subplot,
        units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_convergence(
        tracer, mask=None, extract_array_from_mask=False, zoom_around_mask=False, as_subplot=False,
        units='arcsec', figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Tracer Convergence', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='tracer_convergence'):

    array_plotters.plot_array(
        array=tracer.convergence, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_potential(
        tracer, mask=None, extract_array_from_mask=False, zoom_around_mask=False, as_subplot=False,
       units='arcsec', figsize=(7, 7), aspect='square',
       cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
       title='Tracer Potential', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
       output_path=None, output_format='show', output_filename='tracer_potential'):

    array_plotters.plot_array(
        array=tracer.potential, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_deflections_y(
        tracer, mask=None, extract_array_from_mask=False, zoom_around_mask=False, as_subplot=False,
        units='arcsec', figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Tracer Deflections (y)', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='tracer_deflections_y'):

    array_plotters.plot_array(
        array=tracer.deflections_y, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_deflections_x(
        tracer, mask=None, extract_array_from_mask=False, zoom_around_mask=False, as_subplot=False,
        units='arcsec', figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Tracer Deflections (x)', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        output_path=None, output_format='show', output_filename='tracer_deflections_x'):

    array_plotters.plot_array(
        array=tracer.deflections_x, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)
