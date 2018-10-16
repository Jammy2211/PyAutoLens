from matplotlib import pyplot as plt

from autolens.plotting import tools
from autolens.plotting import plot_array
from autolens.plotting import tools_array
from autolens.plotting import profile_plotters

def plot_intensities(galaxy, grid, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(7, 7), aspect='equal', cmap='jet', 
                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Galaxy Intensities', titlesize=16, xlabelsize=16, ylabelsize=16,
                     output_path=None, output_format='show', output_filename='galaxy_intensities'):

    intensities = galaxy.intensities_from_grid(grid=grid)
    intensities = grid.scaled_array_from_array_1d(intensities)

    plot_array.plot_intensities(intensities=intensities, as_subplot=as_subplot,
                                units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_surface_density(galaxy, grid, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None,
                         xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         figsize=(7, 7), aspect='equal', cmap='jet', 
                         cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                         title='Galaxy Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16,
                         output_path=None, output_format='show', output_filename='galaxy_surface_density'):

    surface_density = galaxy.surface_density_from_grid(grid=grid)
    surface_density = grid.scaled_array_from_array_1d(surface_density)

    plot_array.plot_surface_density(surface_density=surface_density, as_subplot=as_subplot,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                    norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                    cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                    title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                    output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_potential(galaxy, grid, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', 
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Galaxy Potential', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='galaxy_potential'):

    potential = galaxy.potential_from_grid(grid=grid)
    potential = grid.scaled_array_from_array_1d(potential)

    plot_array.plot_potential(potential=potential, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_deflections_y(galaxy, grid, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(7, 7), aspect='equal', cmap='jet', 
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Galaxy Deflections (y)', titlesize=16, xlabelsize=16, ylabelsize=16,
                     output_path=None, output_format='show', output_filename='galaxy_deflections_y'):

    deflections = galaxy.deflections_from_grid(grid)
    deflections_y = grid.scaled_array_from_array_1d(deflections[:,0])

    plot_array.plot_deflections_y(deflections_y=deflections_y, as_subplot=as_subplot,
                                  units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                  norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                  linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                  cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                  title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                  output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_deflections_x(galaxy, grid, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(7, 7), aspect='equal', cmap='jet', 
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Galaxy Deflections (x)', titlesize=16, xlabelsize=16, ylabelsize=16,
                     output_path=None, output_format='show', output_filename='galaxy_deflections_x'):

    deflections = galaxy.deflections_from_grid(grid)
    deflections_x = grid.scaled_array_from_array_1d(deflections[:,1])

    plot_array.plot_deflections_x(deflections_x=deflections_x, as_subplot=as_subplot,
                                  units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                  norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                  linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                  cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                  title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                  output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_intensities_individual(galaxy, grid,
                                units='arcsec', kpc_per_arcsec=None,
                                xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                linscale=0.01,
                                figsize=(7, 7), aspect='equal', cmap='jet',
                                cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                title='Galaxy Intensities', titlesize=10, xlabelsize=10, ylabelsize=10,
                                output_path=None, output_format='show', output_filename='galaxy_individual_intensities'):

    total_light_profiles = len(galaxy.light_profiles)
    rows, columns, figsize = tools.get_subplot_rows_columns_figsize(number_subplots=total_light_profiles)
    plt.figure(figsize=figsize)

    for i, light_profile in enumerate(galaxy.light_profiles):

        plt.subplot(rows, columns, i+1)

        profile_plotters.plot_intensities(light_profile=light_profile, grid=grid, as_subplot=True,
                                          units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                          norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                          linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                          cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                          title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                          output_path=output_path, output_format=output_format,
                                          output_filename=output_filename)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                     output_format=output_format)
    plt.close()
    
def plot_surface_density_individual(galaxy, grid,
                                    units='arcsec', kpc_per_arcsec=None,
                                    xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                    linscale=0.01,
                                    figsize=(7, 7), aspect='equal', cmap='jet',
                                    cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                    title='Galaxy Surface Density', titlesize=10, xlabelsize=10, ylabelsize=10,
                                    output_path=None, output_format='show',
                                    output_filename='galaxy_individual_surface_density'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize = tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)
    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i+1)

        profile_plotters.plot_surface_density(mass_profile=mass_profile, grid=grid, as_subplot=True,
                                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                              output_path=output_path, output_format=output_format,
                                              output_filename=output_filename)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                     output_format=output_format)
    plt.close()
    
def plot_potential_individual(galaxy, grid,
                                    units='arcsec', kpc_per_arcsec=None,
                                    xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                    linscale=0.01,
                                    figsize=(7, 7), aspect='equal', cmap='jet',
                              cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                    title='Galaxy Potential', titlesize=10, xlabelsize=10, ylabelsize=10,
                                    output_path=None, output_format='show',
                                    output_filename='galaxy_individual_potential'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize = tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)
    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i+1)

        profile_plotters.plot_potential(mass_profile=mass_profile, grid=grid, as_subplot=True,
                                        units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                        norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                        linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                        output_path=output_path, output_format=output_format, output_filename=output_filename)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                     output_format=output_format)
    plt.close()
    
def plot_deflections_y_individual(galaxy, grid,
                              units='arcsec', kpc_per_arcsec=None,
                              xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                              linscale=0.01,
                              figsize=(7, 7), aspect='equal', cmap='jet',
                                  cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                              title='Galaxy Deflections (y)', titlesize=10, xlabelsize=10, ylabelsize=10,
                              output_path=None, output_format='show',
                              output_filename='galaxy_individual_deflections_y'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize = tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)
    plt.figure(figsize=figsize)
    
    for i, mass_profile in enumerate(galaxy.mass_profiles):
    
        plt.subplot(rows, columns, i+1)
    
        profile_plotters.plot_deflections_y(mass_profile=mass_profile, grid=grid, as_subplot=True,
                                            units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                            norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                            linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                            title=title, titlesize=titlesize, xlabelsize=xlabelsize,
                                            ylabelsize=ylabelsize,
                                            output_path=output_path, output_format=output_format,
                                            output_filename=output_filename)
    
    tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                     output_format=output_format)
    plt.close()


def plot_deflections_x_individual(galaxy, grid,
                                  units='arcsec', kpc_per_arcsec=None,
                                  xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                  linscale=0.01,
                                  figsize=(7, 7), aspect='equal', cmap='jet',
                                  cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                  title='Galaxy Deflections (x)', titlesize=10, xlabelsize=10, ylabelsize=10,
                                  output_path=None, output_format='show',
                                  output_filename='galaxy_individual_deflections_x'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize = tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)
    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.plot_deflections_x(mass_profile=mass_profile, grid=grid, as_subplot=True,
                                            units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                            norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                            linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap,
                                            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                            title=title, titlesize=titlesize, xlabelsize=xlabelsize,
                                            ylabelsize=ylabelsize,
                                            output_path=output_path, output_format=output_format,
                                            output_filename=output_filename)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                     output_format=output_format)
    plt.close()