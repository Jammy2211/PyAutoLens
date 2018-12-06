from matplotlib import pyplot as plt

from autolens import exc
from autofit import conf
from autolens.model.galaxy import galaxy_data as gd
from autolens.data.array.plotters import plotter_util, array_plotters
from autolens.data.fitting.plotters import fitting_plotters


def plot_single_subplot(fit, should_plot_mask=True, positions=None,
                        units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
                        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                        mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                        output_path=None, output_filename='galaxy_fit', output_format='show', ignore_config=True):

    plot_galaxy_fitting_as_subplot = conf.instance.general.get('output', 'plot_galaxy_fitting_as_subplot', bool)

    if not plot_galaxy_fitting_as_subplot and ignore_config is False:
        return

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=4)

    mask = fitting_plotters.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_galaxy_data_array(galaxy_data=fit.galaxy_datas[0], mask=mask, positions=positions, as_subplot=True,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                           linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           titlesize=titlesize, xlabelsize=xlabelsize,
                           ylabelsize=ylabelsize, xyticksize=xyticksize,
                           grid_pointsize=grid_pointsize, position_pointsize=position_pointsize,
                           mask_pointsize=mask_pointsize,
                           output_path=output_path, output_filename=output_filename,
                           output_format=output_format)

    plt.subplot(rows, columns, 2)

    fitting_plotters.plot_model_image(fit=fit, mask=mask, positions=positions, as_subplot=True,
                                      units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                      cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                      linscale=linscale,
                                      cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                      title='Model Galaxy', titlesize=titlesize, xlabelsize=xlabelsize,
                                      ylabelsize=ylabelsize, xyticksize=xyticksize,
                                      position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
                                      output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 3)

    fitting_plotters.plot_residuals(fit=fit, mask=mask, as_subplot=True,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                    cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                    xyticksize=xyticksize,
                                    position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
                                    output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 4)

    fitting_plotters.plot_chi_squareds(fit=fit, mask=mask, as_subplot=True,
                                       units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                       linscale=linscale,
                                       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                       titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                       xyticksize=xyticksize,
                                       position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
                                       output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                               output_format=output_format)

    plt.close()


def plot_deflections_subplot(fit, should_plot_mask=True, positions=None,
                        units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
                        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                        titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                        mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                        output_path=None, output_filename='galaxy_fit', output_format='show', ignore_config=True):

    plot_galaxy_fitting_as_subplot = conf.instance.general.get('output', 'plot_galaxy_fitting_as_subplot', bool)

    if not plot_galaxy_fitting_as_subplot and ignore_config is False:
        return

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(number_subplots=8)

    mask = fitting_plotters.get_mask(fit=fit, should_plot_mask=should_plot_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_galaxy_data_array(galaxy_data=fit.galaxy_datas[0], mask=mask, positions=positions, as_subplot=True,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                           linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                           grid_pointsize=grid_pointsize, position_pointsize=position_pointsize,
                           mask_pointsize=mask_pointsize,
                           output_path=output_path, output_filename=output_filename,
                           output_format=output_format)


    plt.subplot(rows, columns, 2)

    fitting_plotters.plot_model_image(fit=fit, image_index=0, mask=mask, positions=positions,
                                      as_subplot=True,
                                      units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                      cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                      linscale=linscale,
                                      cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                      title='Model Galaxy', titlesize=titlesize, xlabelsize=xlabelsize,
                                      ylabelsize=ylabelsize, xyticksize=xyticksize,
                                      position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
                                      output_path=output_path, output_filename='', output_format=output_format)
    plt.subplot(rows, columns, 3)

    fitting_plotters.plot_residuals(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                    cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                    xyticksize=xyticksize,
                                    position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
                                    output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 4)

    fitting_plotters.plot_chi_squareds(fit=fit, image_index=0, mask=mask, as_subplot=True,
                                       units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                       linscale=linscale,
                                       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                       titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                       xyticksize=xyticksize,
                                       position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
                                       output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 5)

    plot_galaxy_data_array(galaxy_data=fit.galaxy_datas[1], mask=mask, positions=positions, as_subplot=True,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                           linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
                           grid_pointsize=grid_pointsize, position_pointsize=position_pointsize,
                           mask_pointsize=mask_pointsize,
                           output_path=output_path, output_filename=output_filename,
                           output_format=output_format)


    plt.subplot(rows, columns, 6)

    fitting_plotters.plot_model_image(fit=fit, image_index=1, mask=mask, positions=positions, as_subplot=True,
                                      units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                      cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                      linscale=linscale,
                                      cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                      title='Model Galaxy', titlesize=titlesize, xlabelsize=xlabelsize,
                                      ylabelsize=ylabelsize, xyticksize=xyticksize,
                                      position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
                                      output_path=output_path, output_filename='', output_format=output_format)
    plt.subplot(rows, columns, 7)

    fitting_plotters.plot_residuals(fit=fit, image_index=1, mask=mask, as_subplot=True,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                    cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                    xyticksize=xyticksize,
                                    position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
                                    output_path=output_path, output_filename='', output_format=output_format)

    plt.subplot(rows, columns, 8)

    fitting_plotters.plot_chi_squareds(fit=fit, image_index=1, mask=mask, as_subplot=True,
                                       units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                       cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                       linscale=linscale,
                                       cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                       titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                       xyticksize=xyticksize,
                                       position_pointsize=position_pointsize, mask_pointsize=mask_pointsize,
                                       output_path=output_path, output_filename='', output_format=output_format)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                      output_format=output_format)

    plt.close()

def plot_fitting_individuals(fit, units='kpc', output_path=None, output_format='show'):

    plot_galaxy_fitting_model_image = conf.instance.general.get('output', 'plot_galaxy_fitting_model_image', bool)
    plot_galaxy_fitting_residuals = conf.instance.general.get('output', 'plot_galaxy_fitting_residuals', bool)
    plot_galaxy_fitting_chi_squareds = conf.instance.general.get('output', 'plot_galaxy_fitting_chi_squareds', bool)
    
    mask = fitting_plotters.get_mask(fit=fit, should_plot_mask=True)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec_proper

    if plot_galaxy_fitting_model_image:

        fitting_plotters.plot_model_image(fit=fit, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                          output_path=output_path, output_format=output_format)

    if plot_galaxy_fitting_residuals:

        fitting_plotters.plot_residuals(fit=fit, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                        output_path=output_path, output_format=output_format)

    if plot_galaxy_fitting_chi_squareds:

        fitting_plotters.plot_chi_squareds(fit=fit, mask=mask, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                           output_path=output_path, output_format=output_format)

def plot_galaxy_data_array(galaxy_data, mask=None, positions=None, as_subplot=False,
                           units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
                           cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                           cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                           titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                           mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                           output_path=None, output_filename='galaxy_fit', output_format='show'):

    if isinstance(galaxy_data, gd.GalaxyDataIntensities):

        title='Galaxy Data Intensities'

    elif isinstance(galaxy_data, gd.GalaxyDataSurfaceDensity):

        title='Galaxy Data Surface Density'

    elif isinstance(galaxy_data, gd.GalaxyDataPotential):

        title='Galaxy Data Potential'

    elif isinstance(galaxy_data, gd.GalaxyDataDeflectionsY):

        title='Galaxy Data Deflections (y)'

    elif isinstance(galaxy_data, gd.GalaxyDataDeflectionsX):

        title='Galaxy Data Deflections (x)'

    else:

        raise exc.PlottingException('The galaxy datas supplied to plot_galaxy_data_array is not a supported type')

    array_plotters.plot_array(array=galaxy_data.array, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              grid_pointsize=grid_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)