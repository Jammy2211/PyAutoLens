from matplotlib import pyplot as plt

from autolens import conf
from autolens.plotting import tools
from autolens.plotting import plot_array
from autolens.plotting import plane_plotters

def plot_ray_tracing_subplot(tracer, mask=None, positions=None, units='kpc', output_path=None,
                             output_filename='tracer', output_format='show', ignore_config=True, figsize=(20,14)):
    """Plot the observed _tracer of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    tracer : autolens.imaging.tracer.Image
        Class containing the _tracer, noise-mappers and PSF that are to be plotted.
        The font size of the figure ylabel.
    output_path : str
        The path where the _tracer is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the _tracer is output. File formats (e.g. png, fits) output the _tracer to harddisk. 'show' displays the _tracer \
        in the python interpreter window.
    """

    plot_ray_tracing_as_subplot = conf.instance.general.get('output', 'plot_ray_tracing_as_subplot', bool)

    if plot_ray_tracing_as_subplot or ignore_config is True:

        rows, columns, figsize = tools.get_subplot_rows_columns_figsize(number_subplots=6)
        plt.figure(figsize=figsize)
        plt.subplot(rows, columns, 1)

        plot_image_plane_image(tracer=tracer, mask=mask, positions=positions, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='equal', cmap='jet',
                               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
            title='Image-plane Image', titlesize=10, xlabelsize=10, ylabelsize=10,
            output_path=output_path, output_filename='', output_format=output_format)

        plt.subplot(rows, columns, 2)

        plot_surface_density(tracer=tracer, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='equal', cmap='jet',
                             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
            title='Surface Density', titlesize=10, xlabelsize=10, ylabelsize=10,
            output_path=output_path, output_filename='', output_format=output_format)

        plt.subplot(rows, columns, 3)

        plot_potential(tracer=tracer, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
            title='Gravitational Potential', titlesize=10, xlabelsize=10, ylabelsize=10,
            output_path=output_path, output_filename='', output_format=output_format)

        plt.subplot(rows, columns, 4)

        plane_plotters.plot_plane_image(plane=tracer.source_plane, as_subplot=True,
            positions=None, plot_grid=False,
            units=units, kpc_per_arcsec=tracer.source_plane.kpc_per_arcsec_proper,
            xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='equal', cmap='jet',
                                        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
            title='Source-plane Image', titlesize=10, xlabelsize=10, ylabelsize=10,
            output_path=output_path, output_filename='', output_format=output_format)

        plt.subplot(rows, columns, 5)

        plot_deflections_y(tracer=tracer, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
            title='Deflection Angles (y)', titlesize=10, xlabelsize=10, ylabelsize=10,
            output_path=output_path, output_filename='', output_format=output_format)

        plt.subplot(rows, columns, 6)

        plot_deflections_x(tracer=tracer, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='equal', cmap='jet',
                           cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
            title='Deflection Angles (x)', titlesize=10, xlabelsize=10, ylabelsize=10,
            output_path=output_path, output_filename='', output_format=output_format)

        tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                   output_format=output_format)

        plt.close()

def plot_ray_tracing_individual(tracer, output_path=None, output_format='show'):
    """Plot the observed _tracer of an analysis, using the *Image* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    tracer : autolens.imaging.tracer.Image
        Class containing the _tracer, noise-mappers and PSF that are to be plotted.
        The font size of the figure ylabel.
    output_path : str
        The path where the _tracer is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the _tracer is output. File formats (e.g. png, fits) output the _tracer to harddisk. 'show' displays the _tracer \
        in the python interpreter window.
    """

    plot_ray_tracing_image_plane_image = conf.instance.general.get('output', 'plot_ray_tracing_image_plane_image', bool)
    plot_ray_tracing_source_plane = conf.instance.general.get('output', 'plot_ray_tracing_source_plane_image', bool)
    plot_ray_tracing_surface_density = conf.instance.general.get('output', 'plot_ray_tracing_surface_density', bool)
    plot_ray_tracing_potential = conf.instance.general.get('output', 'plot_ray_tracing_potential', bool)
    plot_ray_tracing_deflections_y = conf.instance.general.get('output', 'plot_ray_tracing_deflections_y', bool)
    plot_ray_tracing_deflections_x = conf.instance.general.get('output', 'plot_ray_tracing_deflections_x', bool)

    if plot_ray_tracing_image_plane_image:

        plot_image_plane_image(tracer=tracer, mask=None, positions=None, output_path=output_path,
                               output_format=output_format)

    if plot_ray_tracing_surface_density:

        plot_surface_density(tracer=tracer, output_path=output_path, output_format=output_format)

    if plot_ray_tracing_potential:

        plot_potential(tracer=tracer, output_path=output_path, output_format=output_format)

    if plot_ray_tracing_source_plane:

        plane_plotters.plot_plane_image(plane=tracer.source_plane, positions=None, plot_grid=False,
            output_path=output_path, output_filename='tracer_source_plane', output_format=output_format)

    if plot_ray_tracing_deflections_y:

        plot_deflections_y(tracer=tracer, output_path=output_path, output_format=output_format)

    if plot_ray_tracing_deflections_x:

        plot_deflections_x(tracer=tracer, output_path=output_path, output_format=output_format)


def plot_image_plane_image(tracer, mask=None, positions=None, grid=None, as_subplot=False,
                           units='arcsec', kpc_per_arcsec=None,
                           xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                           figsize=(7, 7), aspect='equal', cmap='jet',
                           cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                           title='Tracer Image-Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                           output_path=None, output_format='show', output_filename='tracer_image_plane_image'):

    plot_array.plot_image_plane_image(image_plane_image=tracer.image_plane_image, mask=mask, positions=positions,
                                      grid=grid, as_subplot=as_subplot,
                                      units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                      norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                      linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                      cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                      title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                      output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_surface_density(tracer, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None,
                         xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         figsize=(7, 7), aspect='equal', cmap='jet',
                         cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                         title='Tracer Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16,
                         output_path=None, output_format='show', output_filename='tracer_surface_density'):

    plot_array.plot_surface_density(surface_density=tracer.surface_density, as_subplot=as_subplot,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                    norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                    output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_potential(tracer, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet',
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Tracer Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='tracer_potential'):

    plot_array.plot_potential(potential=tracer.potential, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_deflections_y(tracer, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(7, 7), aspect='equal', cmap='jet',
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Tracer Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16,
                       output_path=None, output_format='show', output_filename='tracer_deflections_y'):

    plot_array.plot_deflections_y(deflections_y=tracer.deflections_y, as_subplot=as_subplot,
                                  units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                  norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                  linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                  cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                  title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                  output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_deflections_x(tracer, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(7, 7), aspect='equal', cmap='jet',
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Tracer Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16,
                       output_path=None, output_format='show', output_filename='tracer_deflections_x'):

    plot_array.plot_deflections_x(deflections_x=tracer.deflections_x, as_subplot=as_subplot,
                                  units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                  norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                  linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                  cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                  title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                  output_path=output_path, output_format=output_format, output_filename=output_filename)
