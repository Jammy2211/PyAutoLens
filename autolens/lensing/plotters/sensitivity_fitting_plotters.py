from matplotlib import pyplot as plt

from autolens.data.array.plotters import plotter_util
from autolens.data.imaging.plotters import imaging_plotters
from autolens.data.fitting.plotters import fitting_plotters


def plot_fitting_subplot(fit, should_plot_mask=True, positions=None,
                         units='arcsec', figsize=None, aspect='equal',
                         cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                         linscale=0.01,
                         cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                         titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                         mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                         output_path=None, output_filename='sensitivity_fit', output_format='show'):
    """Plot the model data of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.sensitivity_fitting.SensitivityProfileFit
        Class containing fitting between the model data and observed lensing data (including residuals, chi_squareds etc.)
    """

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=9)

    mask = fitting_plotters.get_mask(fit=fit.fit_normal, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer_normal.image_plane.kpc_per_arcsec_proper

    imaging_plotters.plot_image(image=fit.fit_normal.images[0], mask=mask, positions=positions, image_plane_pix_grid=None,
                                as_subplot=True,
                                units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                linscale=linscale,
                                cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                xyticksize=xyticksize,
                                grid_pointsize=grid_pointsize, position_pointsize=position_pointsize,
                                mask_pointsize=mask_pointsize,
                                output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 4)

    fitting_plotters.plot_model_image(fit=fit.fit_normal, image_index=0, mask=mask, as_subplot=True,
                                      units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                      cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                      linscale=linscale,
                                      cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                      titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                      output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 5)

    fitting_plotters.plot_residuals(fit=fit.fit_normal, image_index=0, mask=mask, as_subplot=True,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                    cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                    output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 6)

    fitting_plotters.plot_chi_squareds(fit=fit.fit_normal, image_index=0, mask=mask, as_subplot=True,
                                       units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                       linscale=linscale,
                                       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                       titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                       output_path=output_path, output_filename='', output_format=output_format)
    
    plt.subplot(rows, columns, 7)

    fitting_plotters.plot_model_image(fit=fit.fit_sensitive, image_index=0, mask=mask, as_subplot=True,
                                      units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                      cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                      linscale=linscale,
                                      cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                      titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                      output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 8)

    fitting_plotters.plot_residuals(fit=fit.fit_sensitive, image_index=0, mask=mask, as_subplot=True,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                    cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                    output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 9)

    fitting_plotters.plot_chi_squareds(fit=fit.fit_sensitive, image_index=0, mask=mask, as_subplot=True,
                                       units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                       linscale=linscale,
                                       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                       titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                                       output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()