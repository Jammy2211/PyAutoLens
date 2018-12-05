from matplotlib import pyplot as plt

from autofit import conf
from autolens.data.array.plotters import plotter_util, array_plotters


def plot_image_subplot(image, plot_origin=True, mask=None, should_plot_border=False, positions=None,
                       image_plane_pix_grid=None,
                       units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
                       cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                       mask_pointsize=10, position_pointsize=30, grid_pointsize=1,
                       output_path=None, output_filename='regular', output_format='show', ignore_config=True):
    """Plot the observed datas_ of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    figsize
    ignore_config
    output_filename
    units
    positions
    image : autolens.imaging.regular.Image
        Class containing the datas_, noise-mappers and PSF that are to be plotted.
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    plot_image_as_subplot = conf.instance.general.get('output', 'plot_imaging_as_subplot', bool)

    if plot_image_as_subplot or ignore_config:

        rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=4)

        if figsize is None:
            figsize = figsize_tool

        plt.figure(figsize=figsize)
        plt.subplot(rows, columns, 1)

        plot_image(image=image, plot_origin=plot_origin, mask=mask, should_plot_border=should_plot_border,
                   positions=positions, image_plane_pix_grid=image_plane_pix_grid, as_subplot=True,
                   units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                   cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
                   cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                   titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                   mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
                   output_path=output_path, output_format=output_format)

        plt.subplot(rows, columns, 2)

        plot_noise_map(image=image, plot_origin=plot_origin, mask=mask, as_subplot=True,
                       units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                       linscale=linscale,
                       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                       titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                       mask_pointsize=mask_pointsize,
                       output_path=output_path, output_format=output_format)

        plt.subplot(rows, columns, 3)

        plot_psf(image=image, as_subplot=True,
                 units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                 cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
                 cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                 titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                 output_path=output_path, output_format=output_format)

        plt.subplot(rows, columns, 4)

        plot_signal_to_noise_map(image=image, plot_origin=plot_origin, mask=mask, as_subplot=True,
                                 units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                 cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                 linscale=linscale,
                                 cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                 titlesize=titlesize, xlabelsize=xlabelsize,
                                 ylabelsize=ylabelsize, xyticksize=xyticksize,
                                 mask_pointsize=mask_pointsize,
                                 output_path=output_path, output_format=output_format)

        plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                          output_format=output_format)

        plt.close()


def plot_image_individual(image, plot_origin=True, mask=None, positions=None, output_path=None, output_format='png'):
    """Plot the observed datas_ of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    figsize
    plot_signal_to_noise_map
    ignore_config
    plot_psf
    plot_noise_map
    plot_image
    positions
    image : autolens.imaging.regular.Image
        Class containing the datas_, noise-mappers and PSF that are to be plotted.
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    plot_imaging_image = conf.instance.general.get('output', 'plot_imaging_image', bool)
    plot_imaging_noise_map = conf.instance.general.get('output', 'plot_imaging_noise_map', bool)
    plot_imaging_psf = conf.instance.general.get('output', 'plot_imaging_psf', bool)
    plot_imaging_signal_to_noise_map = conf.instance.general.get('output', 'plot_imaging_signal_to_noise_map', bool)

    if plot_imaging_image:
        plot_image(image=image, plot_origin=plot_origin, mask=mask, positions=positions, output_path=output_path,
                   output_format=output_format)

    if plot_imaging_noise_map:
        plot_noise_map(image=image,  plot_origin=plot_origin, mask=mask, output_path=output_path,
                       output_format=output_format)

    if plot_imaging_psf:
        plot_psf(image=image, plot_origin=plot_origin, output_path=output_path, output_format=output_format)

    if plot_imaging_signal_to_noise_map:
        plot_signal_to_noise_map(image=image, plot_origin=plot_origin, mask=mask, output_path=output_path,
                                 output_format=output_format)


def plot_image(image, plot_origin=True, mask=None, should_plot_border=False, positions=None,
               image_plane_pix_grid=None, as_subplot=False,
               units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
               title='Observed Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
               mask_pointsize=10, position_pointsize=30, grid_pointsize=1,
               output_path=None, output_format='show', output_filename='observed_image'):

    origin = get_origin(image=image, plot_origin=plot_origin)

    array_plotters.plot_array(array=image, origin=origin, mask=mask, should_plot_border=should_plot_border,
                              positions=positions, grid=image_plane_pix_grid, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              grid_pointsize=grid_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_noise_map(image, plot_origin=True, mask=None, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                   cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                   mask_pointsize=10,
                   output_path=None, output_format='show', output_filename='noise_map_'):

    origin = get_origin(image=image.noise_map, plot_origin=plot_origin)

    array_plotters.plot_array(array=image.noise_map, origin=origin, mask=mask, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_psf(image, plot_origin=True, as_subplot=False,
             units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
             cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
             title='PSF', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
             output_path=None, output_format='show', output_filename='psf'):

    origin = get_origin(image=image.psf, plot_origin=plot_origin)

    array_plotters.plot_array(array=image.psf, origin=origin, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_signal_to_noise_map(image, plot_origin=True, mask=None, as_subplot=False,
                             units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                             cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                             title='Signal-To-Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                             mask_pointsize=10,
                             output_path=None, output_format='show', output_filename='signal_to_noise_map'):

    origin = get_origin(image=image, plot_origin=plot_origin)

    array_plotters.plot_array(array=image.signal_to_noise_map, origin=origin, mask=mask, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def get_origin(image, plot_origin):

    if plot_origin:
        return image.origin
    else:
        return None