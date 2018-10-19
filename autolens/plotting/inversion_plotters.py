from autolens.plotting import tools_array
from autolens.plotting import tools
from autolens.plotting import mapper_plotters

def plot_reconstructed_image(inversion, mask=None, positions=None, grid=None, as_subplot=False,
                             units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                             cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                             title='Reconstructed Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                             output_path=None, output_format='show', output_filename='reconstructed_image'):

    tools_array.plot_array(array=inversion.reconstructed_image, mask=mask, positions=positions, grid=grid,
                           as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_reconstructed_pixelization(inversion, positions=None, should_plot_centres=False,
                                    should_plot_grid=False, image_pixels=None, source_pixels=None, as_subplot=False,
                                    units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                                    cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                    linscale=0.01,
                                    cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                    title='Reconstructed Pixelization', titlesize=16, xlabelsize=16, ylabelsize=16,
                                    xyticksize=16,
                                    output_path=None, output_format='show', output_filename='reconstructed_image'):

    tools.setup_figure(figsize=figsize, as_subplot=as_subplot)

    reconstructed_pixelization = \
        inversion.mapper.reconstructed_pixelization_from_solution_vector(inversion.solution_vector)

    tools_array.plot_array(array=reconstructed_pixelization, positions=positions, as_subplot=True,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           output_filename=output_filename)

    mapper_plotters.plot_rectangular_mapper(mapper=inversion.mapper, should_plot_centres=should_plot_centres,
                                            should_plot_grid=should_plot_grid,
                                            image_pixels=image_pixels, source_pixels=source_pixels,
                                            as_subplot=True,
                                            units=units, kpc_per_arcsec=kpc_per_arcsec,
                                            title=title, titlesize=titlesize, xlabelsize=xlabelsize,
                                            ylabelsize=ylabelsize, xyticksize=xyticksize)

    tools.output_figure(array=reconstructed_pixelization, as_subplot=as_subplot, output_path=output_path,
                        output_filename=output_filename, output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)
