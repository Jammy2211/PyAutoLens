from autolens.plotting import tools_array


def plot_reconstructed_image(inversion, mask=None, positions=None, grid=None, as_subplot=False,
                             units='arcsec', kpc_per_arcsec=None,
                             xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             figsize=(7, 7), aspect='equal', cmap='jet',
                             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                             title='Reconstructed Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                             output_path=None, output_format='show', output_filename='reconstructed_image'):

    tools_array.plot_array(array=inversion.reconstructed_image, mask=mask, positions=positions, grid=grid,
                           as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                           norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                           linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_reconstructed_pixelization(inversion, mask=None, positions=None, grid=None, as_subplot=False,
                                    units='arcsec', kpc_per_arcsec=None,
                                    xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                    linscale=0.01,
                                    figsize=(7, 7), aspect='equal', cmap='jet',
                                    cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                    title='Reconstructed Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                                    output_path=None, output_format='show', output_filename='reconstructed_image'):

    reconstructed_pixelization = \
        inversion.mapper.reconstructed_pixelization_from_solution_vector(inversion.solution_vector)

    tools_array.plot_array(array=reconstructed_pixelization, mask=mask, positions=positions, grid=grid,
                           as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                           norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                           linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           output_path=output_path, output_format=output_format,
                           output_filename=output_filename)

