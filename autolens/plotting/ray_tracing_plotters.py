from matplotlib import pyplot as plt

from autolens import conf
from autolens.plotting import array_plotters

def plot_ray_tracing(tracer, units='kpc', output_path=None, output_filename='tracer', output_format='show',
                     ignore_config=True):
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

        plt.figure(figsize=(25, 20))
        plt.subplot(2, 3, 1)

        array_plotters.plot_array(
            array=tracer.image_plane_image, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xticks=tracer.image_plane_image.xticks, yticks=tracer.image_plane_image.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Image-plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 3, 2)

        array_plotters.plot_array(
            array=tracer.surface_density, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xticks=tracer.surface_density.xticks, yticks=tracer.surface_density.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 3, 3)

        array_plotters.plot_array(
            array=tracer.potential, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xticks=tracer.potential.xticks, yticks=tracer.potential.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Gravitational Potential', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 3, 4)

        plane_image = tracer.plane_images_of_planes(shape=(50, 50))[1]

        array_plotters.plot_array(
            array=plane_image, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.source_plane.kpc_per_arcsec_proper,
            xticks=plane_image.xticks, yticks=plane_image.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Source-plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 3, 5)

        array_plotters.plot_array(
            array=tracer.deflections_x, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xticks=tracer.deflections_x.xticks, yticks=tracer.deflections_x.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Deflection Angles (x)', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        plt.subplot(2, 3, 6)

        array_plotters.plot_array(
            array=tracer.deflections_y, grid=None, as_subplot=True,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xticks=tracer.deflections_y.xticks, yticks=tracer.deflections_y.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Deflection Angles (y)', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename=None, output_format=output_format)

        array_plotters.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                            output_format=output_format)
        plt.close()

def plot_ray_tracing_individual(tracer, units='kpc', plot_image_plane_image=False, plot_surface_density=False, 
                                plot_potential=False, plot_deflections=False, plot_source_plane=False,
                                output_path=None, output_format='show', ignore_config=True):
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

    if not ignore_config:

        plot_image_plane_image = conf.instance.general.get('output', 'plot_ray_tracing_image_plane_image', bool)
        plot_surface_density = conf.instance.general.get('output', 'plot_ray_tracing_surface_density', bool)
        plot_potential = conf.instance.general.get('output', 'plot_ray_tracing_potential', bool)
        plot_deflections = conf.instance.general.get('output', 'plot_ray_tracing_deflections', bool)
        plot_source_plane = conf.instance.general.get('output', 'plot_ray_tracing_source_plane', bool)

    if plot_image_plane_image:

        array_plotters.plot_array(
            array=tracer.image_plane_image, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xticks=tracer.image_plane_image.xticks, yticks=tracer.image_plane_image.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Image-Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename='image_plane_image', output_format=output_format)

    if plot_surface_density:

        array_plotters.plot_array(
            array=tracer.surface_density, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xticks=tracer.surface_density.xticks, yticks=tracer.surface_density.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename='surface_density', output_format=output_format)

    if plot_potential:

        array_plotters.plot_array(
            array=tracer.potential, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xticks=tracer.potential.xticks, yticks=tracer.potential.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Potential', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename='potential', output_format=output_format)

    if plot_source_plane:

        plane_image = tracer.plane_images_of_planes(shape=(50, 50))[1]
    
        array_plotters.plot_array(
            array=plane_image, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=tracer.source_plane.kpc_per_arcsec_proper,
            xticks=plane_image.xticks, yticks=plane_image.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Source Plane', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename='source_plane', output_format=output_format)

    if plot_deflections:

        array_plotters.plot_array(
            array=tracer.deflections_x, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xticks=tracer.deflections_x.xticks, yticks=tracer.deflections_x.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Deflection Angles (x)', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename='deflections_x', output_format=output_format)
    
        array_plotters.plot_array(
            array=tracer.deflections_y, grid=None, as_subplot=False,
            units=units, kpc_per_arcsec=tracer.image_plane.kpc_per_arcsec_proper,
            xticks=tracer.deflections_y.xticks, yticks=tracer.deflections_y.yticks, xyticksize=16,
            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
            figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
            title='Deflection Angles (y)', titlesize=16, xlabelsize=16, ylabelsize=16,
            output_path=output_path, output_filename='deflections_y', output_format=output_format)