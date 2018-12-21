from matplotlib import pyplot as plt

from autofit import conf
from autolens.data.array.plotters import plotter_util
from autolens.lensing.plotters import lensing_plotter_util

def plot_fit_subplot(fit_hyper, fit, should_plot_mask=True, positions=None, should_plot_image_plane_pix=True,
                     units='arcsec', figsize=None, aspect='equal',
                     cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                     mask_pointsize=10, position_pointsize=10, grid_pointsize=1,
                     output_path=None, output_filename='lensing_fit', output_format='show',
                     ignore_config=True):
    
    plot_lensing_fitting_as_subplot = conf.instance.general.get('output', 'plot_lensing_fitting_as_subplot', bool)

    if not plot_lensing_fitting_as_subplot and ignore_config is False:
        return

    if fit.tracer.total_planes == 1:

        plot_fit_subplot_lens_plane_only(fit_hyper=fit_hyper, fit=fit, should_plot_mask=should_plot_mask,
                                         positions=positions,
                                         should_plot_image_plane_pix=should_plot_image_plane_pix,
                                         units=units, figsize=figsize,
                                         aspect=aspect,
                                         cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                                         linthresh=linthresh,
                                         linscale=linscale,
                                         cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                         titlesize=titlesize, xlabelsize=xlabelsize,
                                         ylabelsize=ylabelsize,
                                         xyticksize=xyticksize,
                                         mask_pointsize=mask_pointsize,
                                         position_pointsize=position_pointsize,
                                         grid_pointsize=grid_pointsize,
                                         output_path=output_path, output_filename=output_filename,
                                         output_format=output_format)


def plot_fit_subplot_lens_plane_only(fit_hyper, fit, should_plot_mask=True, positions=None,
                                     should_plot_image_plane_pix=True,
                                     units='arcsec', figsize=None, aspect='equal',
                                     cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                     linscale=0.01,
                                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                     titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                                     mask_pointsize=10, position_pointsize=10, grid_pointsize=1,
                                     output_path=None, output_filename='lensing_hyper_fit', output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model datas_ and observed lensing datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fittings)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fittings) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=9)

    mask = lensing_plotter_util.get_mask(fit=fit_hyper, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    image_plane_pix_grid = lensing_plotter_util.get_image_plane_pix_grid(should_plot_image_plane_pix, fit)

    lensing_plotter_util.plot_image(fit=fit_hyper, mask=mask, positions=positions,
               image_plane_pix_grid=image_plane_pix_grid, as_subplot=True,
               units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
               cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
               linscale=linscale,
               cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
               titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
               xyticksize=xyticksize,
               grid_pointsize=grid_pointsize, position_pointsize=position_pointsize,
               mask_pointsize=mask_pointsize,
               output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 2)

    lensing_plotter_util.plot_model_data(fit=fit_hyper, mask=mask, as_subplot=True,
                                         units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                         cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
                                         cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                         titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                         output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    lensing_plotter_util.plot_residual_map(fit=fit_hyper, mask=mask, as_subplot=True,
                      units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                      cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
                      cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                      titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                      output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 5)

    lensing_plotter_util.plot_chi_squared_map(fit=fit, mask=mask, as_subplot=True,
                         units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                         cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                         linscale=linscale,
                         cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                         titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                         output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.subplot(rows, columns, 4)

    lensing_plotter_util.plot_contribution_maps(fit=fit_hyper, mask=mask, as_subplot=True,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                           linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                           output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 6)

    lensing_plotter_util.plot_chi_squared_map(fit=fit_hyper, mask=mask, as_subplot=True,
                         units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                         cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                         linscale=linscale,
                         cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                         titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                         output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 8)

    lensing_plotter_util.plot_noise_map(fit=fit, mask=mask, as_subplot=True,
                   units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                   cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                   linscale=linscale,
                   cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                   titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                   xyticksize=xyticksize,
                   output_path=output_path, output_format=output_format)

    plt.subplot(rows, columns, 9)

    lensing_plotter_util.plot_noise_map(fit=fit_hyper, mask=mask, as_subplot=True,
                   units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                   cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                   linscale=linscale,
                   cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                   titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                   output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()


def plot_fit_individuals(fit_hyper, fit, units='kpc', output_path=None, output_format='show'):
    
    if fit.tracer.total_planes == 1:

        plot_fit_individuals_lens_plane_only(fit_hyper=fit_hyper, fit=fit, units=units, 
                                             output_path=output_path, output_format=output_format)

def plot_fit_individuals_lens_plane_only(fit_hyper, fit, units='kpc', output_path=None, output_format='show'):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lensing.fittingting.Fitter
        Class containing fitting between the model datas_ and observed lensing datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fittings)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fittings) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    plot_lensing_fitting_model_image = \
        conf.instance.general.get('output', 'plot_lensing_fitting_model_image', bool)
    plot_lensing_fitting_residuals = \
        conf.instance.general.get('output', 'plot_lensing_fitting_residuals', bool)
    plot_lensing_fitting_chi_squareds = \
        conf.instance.general.get('output', 'plot_lensing_fitting_chi_squareds', bool)
    plot_lensing_fitting_contributions = \
        conf.instance.general.get('output', 'plot_lensing_fitting_contributions', bool)
    plot_lensing_fitting_hyper_chi_squareds = \
        conf.instance.general.get('output', 'plot_lensing_fitting_hyper_chi_squareds', bool)
    plot_lensing_fitting_hyper_noise_map = \
        conf.instance.general.get('output', 'plot_lensing_fitting_hyper_noise_map', bool)

    mask = lensing_plotter_util.get_mask(fit=fit, should_plot_mask=True)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    if plot_lensing_fitting_model_image:
        lensing_plotter_util.plot_model_data(fit=fit, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                             output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_residuals:
        lensing_plotter_util.plot_residual_map(fit=fit, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                          output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_chi_squareds:
        lensing_plotter_util.plot_chi_squared_map(fit=fit, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                             output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_contributions:
        lensing_plotter_util.plot_contribution_maps(fit=fit, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                               output_path=output_path, output_format=output_format)

    if plot_lensing_fitting_hyper_noise_map:
        lensing_plotter_util.plot_noise_map(fit=fit_hyper, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                       output_path=output_path, output_filename='fit_hyper_noise_map', output_format=output_format)

    if plot_lensing_fitting_hyper_chi_squareds:
        lensing_plotter_util.plot_chi_squareds(fit=fit_hyper, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                          output_path=output_path, output_filename='fit_hyper_chi_squared_map',
                          output_format=output_format)