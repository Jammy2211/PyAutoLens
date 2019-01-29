from autolens.data.array.plotters import array_plotters

import numpy as np

def plot_image(fit, mask=None, positions=None, image_plane_pix_grid=None, as_subplot=False,
               units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
               title='Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
               grid_pointsize=1, mask_pointsize=10, position_pointsize=10,
               output_path=None, output_format='show', output_filename='fit_image'):
    """Plot the image of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.ccd.datas.CCD
        The datas-datas, which includes the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    masked_image = np.add(fit.image, 0.0, out=np.zeros_like(fit.image), where=np.asarray(mask) == 0)

    array_plotters.plot_array(masked_image, mask=mask, grid=image_plane_pix_grid,
                              positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              grid_pointsize=grid_pointsize, mask_pointsize=mask_pointsize,
                              position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_noise_map(fit, mask=None, positions=None, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                   cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                   mask_pointsize=10, position_pointsize=10,
                   output_path=None, output_format='show', output_filename='fit_noise_map'):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    image : datas.ccd.datas.CCD
        The datas-datas, which includes the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    plot_origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    array_plotters.plot_array(array=fit.noise_map, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_model_data(fit, mask=None, positions=None, as_subplot=False,
                    units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                    cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                    cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                    title='Fit Model Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                    mask_pointsize=10, position_pointsize=10,
                    output_path=None, output_format='show', output_filename='fit_model_image'):
    """Plot the model image of a fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    """
    array_plotters.plot_array(array=fit.model_data, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_model_image_of_plane(fit, plane_index=0, mask=None, positions=None, as_subplot=False,
                              units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                              cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                              cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                              title='Fit Model Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                              mask_pointsize=10, position_pointsize=10,
                              output_path=None, output_format='show', output_filename='fit_model_image_of_plane'):
    """Plot the model image of a specific plane of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    plane_index : int
        The plane from which the model image is generated.
    """
    array_plotters.plot_array(array=fit.model_image_of_planes[plane_index], mask=mask, positions=positions,
                              as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_residual_map(fit, mask=None, positions=None, as_subplot=False,
                      units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                      cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                      cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                      title='Fit Residuals', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                      mask_pointsize=10, position_pointsize=10,
                      output_path=None, output_format='show', output_filename='fit_residual_map'):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """
    array_plotters.plot_array(array=fit.residual_map, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_chi_squared_map(fit, mask=None, positions=None, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                         cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                         title='Fit Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                         mask_pointsize=10, position_pointsize=10,
                         output_path=None, output_format='show', output_filename='fit_chi_squared_map'):
    """Plot the chi-squared map of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the chi-squareds are plotted.
    """
    array_plotters.plot_array(array=fit.chi_squared_map, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_contribution_maps(fit, mask=None, positions=None, as_subplot=False,
                           units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                           cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                           cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                           title='Contributions', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                           mask_pointsize=10, position_pointsize=10,
                           output_path=None, output_format='show', output_filename='fit_contribution_maps'):
    """Plot the summed contribution maps of a hyper-fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The hyper-fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the contribution_maps are plotted.
    """

    if len(fit.contribution_maps) > 1:
        contribution_map = sum(fit.contribution_maps)
    else:
        contribution_map = fit.contribution_maps[0]

    array_plotters.plot_array(array=contribution_map, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def get_image_plane_pix_grid(should_plot_image_plane_pix, fit):

    if hasattr(fit, 'mapper'):
        if should_plot_image_plane_pix and fit.mapper.is_image_plane_pixelization:
            return fit.tracer.image_plane.grids[0].pix
    else:
        return None

def get_mask(fit, should_plot_mask):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    should_plot_mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if should_plot_mask:
        return fit.mask
    else:
        return None