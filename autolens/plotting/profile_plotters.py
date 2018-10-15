from autolens.plotting import plot_array

def plot_intensities(light_profile, grid, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                     title='Intensities', titlesize=16, xlabelsize=16, ylabelsize=16,
                     output_path=None, output_format='show', output_filename='intensities'):

    intensities = light_profile.intensities_from_grid(grid=grid)
    intensities = grid.scaled_array_from_array_1d(intensities)

    plot_array.plot_intensities(intensities=intensities, as_subplot=as_subplot,
                                units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                                title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_surface_density(mass_profile, grid, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None,
                         xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                         title='Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16,
                         output_path=None, output_format='show', output_filename='surface_density'):

    surface_density = mass_profile.surface_density_from_grid(grid=grid)
    surface_density = grid.scaled_array_from_array_1d(surface_density)

    plot_array.plot_surface_density(surface_density=surface_density, as_subplot=as_subplot,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                    norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                                    title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                    output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_potential(mass_profile, grid, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Potential', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='potential'):

    potential = mass_profile.potential_from_grid(grid=grid)
    potential = grid.scaled_array_from_array_1d(potential)

    plot_array.plot_potential(potential=potential, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_deflections_y(mass_profile, grid, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                       title='Deflections (y)', titlesize=16, xlabelsize=16, ylabelsize=16,
                       output_path=None, output_format='show', output_filename='deflections_y'):
    
    deflections = mass_profile.deflections_from_grid(grid)
    deflections_y = grid.scaled_array_from_array_1d(deflections[:,0])

    plot_array.plot_deflections_y(deflections_y=deflections_y, as_subplot=as_subplot,
                                  units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                  norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                  linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                                  title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                  output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_deflections_x(mass_profile, grid, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                       linscale=0.01,
                       figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                       title='Deflections (x)', titlesize=16, xlabelsize=16, ylabelsize=16,
                       output_path=None, output_format='show', output_filename='deflections_x'):

    deflections = mass_profile.deflections_from_grid(grid)
    deflections_x = grid.scaled_array_from_array_1d(deflections[:, 1])

    plot_array.plot_deflections_x(deflections_x=deflections_x, as_subplot=as_subplot,
                                  units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                  norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                  linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                                  title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                  output_path=output_path, output_format=output_format, output_filename=output_filename)