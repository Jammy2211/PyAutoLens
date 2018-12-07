from autolens.data.array.plotters import array_plotters

def plot_model_image(fit, image_index=0, mask=None, positions=None, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                     cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Fit Model Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                     mask_pointsize=10, position_pointsize=10.0,
                     output_path=None, output_format='show', output_filename='fit_model_image'):
    """Plot the model-image of a fit.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    fit : data.fitting.fitting.AbstractFit
        The fit to the data, which includes a list of every model-image, residuals, chi-squareds, etc.
    image_index : int
        The index of the image in the data-set of which the model-image is plotted.
    """
    array_plotters.plot_array(array=fit.model_datas[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_model_image_of_plane(fit, image_index=0, plane_index=0, mask=None, positions=None, as_subplot=False,
                              units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                              cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                              cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                              title='Fit Model Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                              mask_pointsize=10, position_pointsize=10.0,
                              output_path=None, output_format='show', output_filename='fit_model_image_of_plane'):
    """Plot the model-image of a specific plane of a lensing fit.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    fit : data.fitting.fitting.AbstractFit
        The fit to the data, which includes a list of every model-image, residuals, chi-squareds, etc.
    image_index : int
        The index of the image in the data-set of which the model-image is plotted.
    plane_index : int
        The plane from which the model-image is generated.
    """
    array_plotters.plot_array(array=fit.model_images_of_planes[image_index][plane_index], mask=mask, positions=positions,
                              as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_residuals(fit, image_index=0, mask=None, positions=None, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                   cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Fit Residuals', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                   mask_pointsize=10, position_pointsize=10.0,
                   output_path=None, output_format='show', output_filename='fit_residuals'):
    """Plot the residuals of a fit.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    fit : data.fitting.fitting.AbstractFit
        The fit to the data, which includes a list of every model-image, residuals, chi-squareds, etc.
    image_index : int
        The index of the image in the data-set of which the residuals are plotted.
    """
    array_plotters.plot_array(array=fit.residuals[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_chi_squareds(fit, image_index=0, mask=None, positions=None, as_subplot=False,
                      units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                      cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                      cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                      title='Fit Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                      mask_pointsize=10, position_pointsize=10.0,
                      output_path=None, output_format='show', output_filename='fit_chi_squareds'):
    """Plot the chi-squareds of a fit.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    fit : data.fitting.fitting.AbstractFit
        The fit to the data, which includes a list of every model-image, residuals, chi-squareds, etc.
    image_index : int
        The index of the image in the data-set of which the chi-squareds are plotted.
    """
    array_plotters.plot_array(array=fit.chi_squareds[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_contributions(fit, image_index=0, mask=None, positions=None, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                       cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Contributions', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                       mask_pointsize=10, position_pointsize=10.0,
                       output_path=None, output_format='show', output_filename='fit_contributions'):
    """Plot the summed contribution maps of a hyper-fit.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    fit : data.fitting.fitting.AbstractHyperFit
        The hyper-fit to the data, which includes a list of every model-image, residuals, chi-squareds, etc.
    image_index : int
        The index of the image in the data-set of which the contributions are plotted.
    """
    if len(fit.contributions[image_index]) > 1:
        contributions = sum(fit.contributions[image_index])
    else:
        contributions = fit.contributions[image_index][0]

    array_plotters.plot_array(array=contributions, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_scaled_model_image(fit, image_index=0, mask=None, positions=None, as_subplot=False,
                            units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                            cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                            cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                            title='Fit Scaled Model Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                            mask_pointsize=10, position_pointsize=10.0,
                            output_path=None, output_format='show', output_filename='fit_scaled_model_image'):
    """Plot the scaled model image of a hyper-fit.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    fit : data.fitting.fitting.AbstractHyperFit
        The hyper-fit to the data, which includes a list of every model-image, residuals, chi-squareds, etc.
    image_index : int
        The index of the image in the data-set of which the scaled model image is plotted.
    """
    array_plotters.plot_array(array=fit.scaled_model_images[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_scaled_residuals(fit, image_index=0, mask=None, positions=None, as_subplot=False,
                          units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                          cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                          cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                          title='Fit Scaled Residuals', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                          mask_pointsize=10, position_pointsize=10.0,
                          output_path=None, output_format='show', output_filename='fit_scaled_residuals'):
    """Plot the scaled residuals of a hyper-fit.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    fit : data.fitting.fitting.AbstractHyperFit
        The hyper-fit to the data, which includes a list of every model-image, residuals, chi-squareds, etc.
    image_index : int
        The index of the image in the data-set of which the scaled residuals are plotted.
    """
    array_plotters.plot_array(array=fit.scaled_residuals[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_scaled_chi_squareds(fit, image_index=0, mask=None, positions=None, as_subplot=False,
                             units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                             cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                             title='Fit Scaled Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                             mask_pointsize=10, position_pointsize=10.0,
                             output_path=None, output_format='show', output_filename='fit_scaled_chi_squareds'):
    """Plot the scaled chi-squareds of a hyper-fit.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    fit : data.fitting.fitting.AbstractHyperFit
        The hyper-fit to the data, which includes a list of every model-image, residuals, chi-squareds, etc.
    image_index : int
        The index of the image in the data-set of which the scaled chi-squareds are plotted.
    """
    array_plotters.plot_array(array=fit.scaled_chi_squareds[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_scaled_noise_map(fit, image_index=0, mask=None, positions=None, as_subplot=False,
                          units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                          cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                          cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                          title='Fit Scaled Noise Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                          mask_pointsize=10, position_pointsize=10.0,
                          output_path=None, output_format='show', output_filename='fit_scaled_noise_map'):
    """Plot the scaled noise-map of a hyper-fit.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    fit : data.fitting.fitting.AbstractHyperFit
        The hyper-fit to the data, which includes a list of every model-image, residuals, chi-squareds, etc.
    image_index : int
        The index of the image in the data-set of which the scaled noise-map is plotted.
    """
    array_plotters.plot_array(array=fit.scaled_noise_maps[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


# def plot_fitting_hyper_arrays(fitting, output_path=None, output_format='show'):
#
#     plot_fitting_hyper_arrays = conf.instance.general.get('output', 'plot_fitting_hyper_arrays', bool)
#
#     if plot_fitting_hyper_arrays:
#
#         array_plotters.plot_array(fitting.unmasked_model_image, output_filename='unmasked_model_image',
#                                         output_path=output_path, output_format=output_format)
#
#         for i, unmasked_galaxy_model_image in enumerate(fitting.unmasked_model_images_of_galaxies):
#             array_plotters.plot_array(unmasked_galaxy_model_image,
#                                             output_filename='unmasked_galaxy_image_' + str(i),
#                                             output_path=output_path, output_format=output_format)

def get_mask(fit, should_plot_mask):
    """Get the mask of the fit if the mask should be plotted on the fit.

    Parameters
    -----------
    fit : data.fitting.fitting.AbstractHyperFit
        The fit to the data, which includes a list of every model-image, residuals, chi-squareds, etc.
    should_plot_mask : bool
        If *True*, the mask is plotted on the fit's image.
    """
    if should_plot_mask:
        return fit.masks[0]
    else:
        return None