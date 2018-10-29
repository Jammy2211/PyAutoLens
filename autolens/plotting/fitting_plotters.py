from matplotlib import pyplot as plt

from autolens.plotting import tools_array

def plot_model_image(fit, image_index=0, mask=None, positions=None, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                     cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Fit Model Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                     mask_pointsize=10, position_pointsize=10.0,
                     output_path=None, output_format='show', output_filename='fit_model_image'):

    tools_array.plot_array(array=fit.model_datas[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
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

    tools_array.plot_array(array=fit.model_images_of_planes[image_index][plane_index], mask=mask, positions=positions,
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

    tools_array.plot_array(array=fit.residuals[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
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

    tools_array.plot_array(array=fit.chi_squareds[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
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

    if len(fit.contributions[image_index]) > 1:
        contributions = sum(fit.contributions[image_index])
    else:
        contributions = fit.contributions[image_index][0]

    tools_array.plot_array(array=contributions, mask=mask, positions=positions, as_subplot=as_subplot,
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

    tools_array.plot_array(array=fit.scaled_model_images[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
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

    tools_array.plot_array(array=fit.scaled_residuals[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
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

    tools_array.plot_array(array=fit.scaled_chi_squareds[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
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

    tools_array.plot_array(array=fit.scaled_noise_maps[image_index], mask=mask, positions=positions, as_subplot=as_subplot,
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

    if should_plot_mask:
        return fit.masks[0]
    else:
        return None