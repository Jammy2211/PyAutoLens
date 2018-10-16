from autolens.plotting import tools
from autolens.plotting import tools_array

def plot_image(image, mask=None, positions=None, grid=None, as_subplot=False,
               units='arcsec', kpc_per_arcsec=None,
               xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
               figsize=(7, 7), aspect='equal', cmap='jet', 
               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
               title='Observed Image', titlesize=16, xlabelsize=16, ylabelsize=16,
               output_path=None, output_format='show', output_filename='observed_image'):

    if positions is not None:
        positions = list(map(lambda pos: image.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=pos), positions))

    tools_array.plot_array(image, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh, linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=image.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=image.xticks, yticks=image.yticks, xlabelsize=xlabelsize,
                                                  ylabelsize=ylabelsize, xyticksize=xyticksize)

    # TODO : if you use set_colorbar and plt.scatter the scatter plot doesnt show...default to removing colorbar now

    if mask is None:
        tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools_array.plot_mask(mask)
    tools_array.plot_points(positions)
    tools_array.plot_grid(grid)
    tools.output_figure(image, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename, output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_noise_map(noise_map, mask=None, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet',
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='noise_map'):


    tools_array.plot_array(noise_map, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh, linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=noise_map.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=noise_map.xticks, yticks=noise_map.yticks,
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools_array.plot_mask(mask)
    tools.output_figure(noise_map, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_psf(psf, as_subplot=False,
             units='arcsec', kpc_per_arcsec=None,
             xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
             figsize=(7, 7), aspect='equal', cmap='jet',
             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
             title='PSF', titlesize=16, xlabelsize=16, ylabelsize=16,
             output_path=None, output_format='show', output_filename='psf'):

    tools_array.plot_array(psf, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm,
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh, linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=psf.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=psf.xticks, yticks=psf.yticks,
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(psf, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename, 
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_signal_to_noise_map(signal_to_noise_map, mask=None, as_subplot=False,
                             units='arcsec', kpc_per_arcsec=None,
                             xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             figsize=(7, 7), aspect='equal', cmap='jet',
                             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                             title='Noise-Map', titlesize=16, xlabelsize=16, ylabelsize=16,
                             output_path=None, output_format='show', output_filename='signal_to_noise_map'):


    tools_array.plot_array(signal_to_noise_map, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, 
                           norm=norm, norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=signal_to_noise_map.shape, units=units,
                                                  kpc_per_arcsec=kpc_per_arcsec,
                                                  xticks=signal_to_noise_map.xticks, yticks=signal_to_noise_map.yticks,
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools_array.plot_mask(mask)
    tools.output_figure(signal_to_noise_map, as_subplot=as_subplot, output_path=output_path, 
                        output_filename=output_filename, output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_intensities(intensities, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(7, 7), aspect='equal', cmap='jet',
                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Intensities', titlesize=16, xlabelsize=16, ylabelsize=16,
                     output_path=None, output_format='show', output_filename='intensities'):

    tools_array.plot_array(intensities, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=intensities.shape, units=units,
                                                  kpc_per_arcsec=kpc_per_arcsec,
                                                  xticks=intensities.xticks, yticks=intensities.yticks,
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(intensities, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename, 
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)


def plot_surface_density(surface_density, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None,
                         xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         figsize=(7, 7), aspect='equal', cmap='jet',
                         cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                         title='Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16,
                         output_path=None, output_format='show', output_filename='surface_density'):
    tools_array.plot_array(surface_density, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=surface_density.shape, units=units,
                                                  kpc_per_arcsec=kpc_per_arcsec,
                                                  xticks=surface_density.xticks, yticks=surface_density.yticks, 
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(surface_density, as_subplot=as_subplot, output_path=output_path,
                        output_filename=output_filename, output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)


def plot_potential(potential, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet',
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Potential', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='potential'):
    tools_array.plot_array(potential, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=potential.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=potential.xticks, yticks=potential.yticks,
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(potential, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename, 
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)


def plot_deflections_y(deflections_y, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(7, 7), aspect='equal', cmap='jet',
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Deflections (y)', titlesize=16, xlabelsize=16, ylabelsize=16,
                       output_path=None, output_format='show', output_filename='deflections_y'):
    tools_array.plot_array(deflections_y, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=deflections_y.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=deflections_y.xticks, yticks=deflections_y.yticks,
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(deflections_y, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)


def plot_deflections_x(deflections_x, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(7, 7), aspect='equal', cmap='jet',
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Deflections (x)', titlesize=16, xlabelsize=16, ylabelsize=16,
                       output_path=None, output_format='show', output_filename='deflections_x'):

    tools_array.plot_array(deflections_x, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=deflections_x.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=deflections_x.xticks, yticks=deflections_x.yticks, 
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(deflections_x, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename, 
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)


def plot_image_plane_image(image_plane_image, mask=None, positions=None, grid=None, as_subplot=False,
                           units='arcsec', kpc_per_arcsec=None,
                           xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                           figsize=(7, 7), aspect='equal', cmap='jet',
                           cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                           title='Image-Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                           output_path=None, output_format='show', output_filename='plane_image_plane_image'):

    if positions is not None:
        positions = list(map(lambda pos: image_plane_image.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=pos),
                             positions))

    tools_array.plot_array(image_plane_image, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, 
                           norm=norm, norm_max=norm_max, norm_min=norm_min,
                           linthresh=linthresh, linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=image_plane_image.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=image_plane_image.xticks, yticks=image_plane_image.yticks, 
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools_array.plot_points(positions)
    tools_array.plot_mask(mask)
    tools_array.plot_grid(grid)
    tools.output_figure(image_plane_image, as_subplot=as_subplot, output_path=output_path, 
                        output_filename=output_filename, output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_plane_image(plane_image, positions=None, plot_grid=False, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(7, 7), aspect='equal', cmap='jet',
                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                     output_path=None, output_format='show', output_filename='plane_image'):

    if positions is not None:
        positions = list(map(lambda pos: plane_image.grid_arc_seconds_to_grid_pixels(grid_arc_seconds=pos),
                             positions))

    tools_array.plot_array(plane_image, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min,
                           linthresh=linthresh, linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=plane_image.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=plane_image.xticks, yticks=plane_image.yticks,
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools_array.plot_points(positions)
    if plot_grid:
        tools_array.plot_grid(plane_image.grid)
    tools.output_figure(plane_image, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_model_image(model_image, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(7, 7), aspect='equal', cmap='jet',
                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                     output_path=None, output_format='show', output_filename='model_image'):

    tools_array.plot_array(model_image, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=model_image.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=model_image.xticks, yticks=model_image.yticks, 
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(model_image, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename, 
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_residuals(residuals, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet',
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Residuals', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='residuals'):

    tools_array.plot_array(residuals, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=residuals.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=residuals.xticks, yticks=residuals.yticks,
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(residuals, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename, 
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)
    
def plot_chi_squareds(chi_squareds, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(7, 7), aspect='equal', cmap='jet',
                      cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
                     output_path=None, output_format='show', output_filename='chi_squareds'):

    tools_array.plot_array(chi_squareds, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=chi_squareds.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=chi_squareds.xticks, yticks=chi_squareds.yticks, 
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(chi_squareds, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename, 
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_contributions(contributions, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(7, 7), aspect='equal', cmap='jet',
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Scaled Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                       output_path=None, output_format='show', output_filename='contributions'):

    tools_array.plot_array(contributions, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, 
                           norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=contributions.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=contributions.xticks, yticks=contributions.yticks, 
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(contributions, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename, 
                        output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)

def plot_scaled_model_image(scaled_model_image, as_subplot=False,
                            units='arcsec', kpc_per_arcsec=None,
                            xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                            figsize=(7, 7), aspect='equal', cmap='jet',
                            cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                            title='Scaled Model Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                            output_path=None, output_format='show', output_filename='scaled_model_image'):

    tools_array.plot_array(scaled_model_image, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, 
                           norm=norm, norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=scaled_model_image.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=scaled_model_image.xticks, yticks=scaled_model_image.yticks, 
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(scaled_model_image, as_subplot=as_subplot, output_path=output_path,
                        output_filename=output_filename, output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)


def plot_scaled_residuals(scaled_residuals, as_subplot=False,
                          units='arcsec', kpc_per_arcsec=None,
                          xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                          figsize=(7, 7), aspect='equal', cmap='jet',
                          cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                          title='Scaled Residuals', titlesize=16, xlabelsize=16, ylabelsize=16,
                          output_path=None, output_format='show', output_filename='scaled_residuals'):

    tools_array.plot_array(scaled_residuals, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, 
                           norm=norm, norm_max=norm_max, norm_min=norm_min, linthresh=linthresh, linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=scaled_residuals.shape, units=units, kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=scaled_residuals.xticks, yticks=scaled_residuals.yticks,
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(scaled_residuals, as_subplot=as_subplot, output_path=output_path, 
                        output_filename=output_filename, output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)


def plot_scaled_chi_squareds(scaled_chi_squareds, as_subplot=False,
                             units='arcsec', kpc_per_arcsec=None,
                             xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             figsize=(7, 7), aspect='equal', cmap='jet',
                             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                             title='Scaled Chi-Squareds', titlesize=16, xlabelsize=16, ylabelsize=16,
                             output_path=None, output_format='show', output_filename='scaled_chi_squareds'):

    tools_array.plot_array(scaled_chi_squareds, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, 
                           norm=norm, norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                           linscale=linscale)
    tools.set_title(title=title, titlesize=titlesize)
    tools_array.set_xy_labels_and_ticks_in_pixels(shape=scaled_chi_squareds.shape, units=units, 
                                                  kpc_per_arcsec=kpc_per_arcsec, 
                                                  xticks=scaled_chi_squareds.xticks, yticks=scaled_chi_squareds.yticks, 
                                                  xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
    tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
    tools.output_figure(scaled_chi_squareds, as_subplot=as_subplot, output_path=output_path, 
                        output_filename=output_filename, output_format=output_format)
    tools.close_figure(as_subplot=as_subplot)
    
def plot_scaled_noise_map(scaled_noise_map, as_subplot=False,
                          units='arcsec', kpc_per_arcsec=None,
                          xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                          figsize=(7, 7), aspect='equal', cmap='jet',
                          cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                          title='Scaled Noise Map', titlesize=16, xlabelsize=16, ylabelsize=16,
                          output_path=None, output_format='show', output_filename='scaled_noise_map'):
    
        tools_array.plot_array(scaled_noise_map, as_subplot=as_subplot, figsize=figsize, aspect=aspect, cmap=cmap, norm=norm, norm_max=norm_max, norm_min=norm_min, linthresh=linthresh,
                               linscale=linscale)
        tools.set_title(title=title, titlesize=titlesize)
        tools_array.set_xy_labels_and_ticks_in_pixels(shape=scaled_noise_map.shape, units=units, 
                                                      kpc_per_arcsec=kpc_per_arcsec, 
                                                      xticks=scaled_noise_map.xticks, yticks=scaled_noise_map.yticks, 
                                                      xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)
        tools_array.set_colorbar(cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad)
        tools.output_figure(scaled_noise_map, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename, output_format=output_format)
        tools.close_figure(as_subplot=as_subplot)