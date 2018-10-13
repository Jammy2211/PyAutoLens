from matplotlib import pyplot as plt

from autolens.plotting import plotters
from autolens.plotting import plotter_tools
from autolens.plotting import profile_plotters

def plot_intensities(galaxy, grid, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                     title='Galaxy Intensities', titlesize=46, xlabelsize=36, ylabelsize=36,
                     output_path=None, output_format='show', output_filename='galaxy_intensities'):

    intensities = galaxy.intensities_from_grid(grid=grid)
    intensities = grid.scaled_array_from_array_1d(intensities)

    plotters.plot_intensities(intensities, as_subplot,
                              units, kpc_per_arcsec, xyticksize, norm, norm_min, norm_max,
                              linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title, titlesize,
                              xlabelsize, ylabelsize, output_path, output_format, output_filename)


def plot_surface_density(galaxy, grid, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None,
                         xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                         title='Galaxy Surface Density', titlesize=46, xlabelsize=36, ylabelsize=36,
                         output_path=None, output_format='show', output_filename='galaxy_surface_density'):

    surface_density = galaxy.surface_density_from_grid(grid=grid)
    surface_density = grid.scaled_array_from_array_1d(surface_density)

    plotters.plot_surface_density(surface_density, as_subplot,
                                  units, kpc_per_arcsec, xyticksize, norm, norm_min, norm_max,
                                  linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title, titlesize,
                                  xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_potential(galaxy, grid, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                   title='Galaxy Potential', titlesize=46, xlabelsize=36, ylabelsize=36,
                   output_path=None, output_format='show', output_filename='galaxy_potential'):

    potential = galaxy.potential_from_grid(grid=grid)
    potential = grid.scaled_array_from_array_1d(potential)

    plotters.plot_potential(potential, as_subplot,
                            units, kpc_per_arcsec, xyticksize, norm, norm_min, norm_max,
                            linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title, titlesize,
                            xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_deflections_y(galaxy, grid, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                     title='Galaxy Deflections (y)', titlesize=46, xlabelsize=36, ylabelsize=36,
                     output_path=None, output_format='show', output_filename='galaxy_deflections_y'):

    deflections = galaxy.deflections_from_grid(grid)
    deflections_y = grid.scaled_array_from_array_1d(deflections[:,0])

    plotters.plot_deflections_y(deflections_y, as_subplot,
                                units, kpc_per_arcsec, xyticksize, norm, norm_min,
                                norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                                titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_deflections_x(galaxy, grid, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                     title='Galaxy Deflections (x)', titlesize=46, xlabelsize=36, ylabelsize=36,
                     output_path=None, output_format='show', output_filename='galaxy_deflections_x'):

    deflections = galaxy.deflections_from_grid(grid)
    deflections_x = grid.scaled_array_from_array_1d(deflections[:,1])

    plotters.plot_deflections_x(deflections_x, units, as_subplot,
                                kpc_per_arcsec, xyticksize, norm, norm_min,
                                norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                                titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_intensities_individual(galaxy, grid,
                                units='arcsec', kpc_per_arcsec=None,
                                xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                linscale=0.01,
                                figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                                title='Galaxy Intensities', titlesize=46, xlabelsize=36, ylabelsize=36,
                                output_path=None, output_format='show', output_filename='galaxy_individual_intensities'):

    total_light_profiles = len(galaxy.light_profiles)
    rows, columns, figsize = plotter_tools.get_subplot_rows_columns_figsize(number_subplots=total_light_profiles)
    plt.figure(figsize=figsize)
    as_subplot = True

    for i, light_profile in enumerate(galaxy.light_profiles):

        plt.subplot(rows, columns, i+1)

        profile_plotters.plot_intensities(light_profile, grid, as_subplot,
                                          units, kpc_per_arcsec, xyticksize, norm,
                                          norm_min, norm_max, linthresh, linscale, figsize, aspect, cmap,
                                          cb_ticksize, title, titlesize, xlabelsize, ylabelsize,
                                          output_path, output_format, output_filename)

    plotter_tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                       output_format=output_format)
    plt.close()
    
def plot_surface_density_individual(galaxy, grid,
                                    units='arcsec', kpc_per_arcsec=None,
                                    xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                    linscale=0.01,
                                    figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                                    title='Galaxy Surface Density', titlesize=46, xlabelsize=36, ylabelsize=36,
                                    output_path=None, output_format='show',
                                    output_filename='galaxy_individual_surface_density'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize = plotter_tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)
    plt.figure(figsize=figsize)
    as_subplot = True

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i+1)

        profile_plotters.plot_surface_density(mass_profile, grid, as_subplot,
                                              units, kpc_per_arcsec, xyticksize, norm,
                                              norm_min, norm_max, linthresh, linscale, figsize, aspect, cmap,
                                              cb_ticksize, title, titlesize, xlabelsize, ylabelsize,
                                              output_path, output_format, output_filename)

    plotter_tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                       output_format=output_format)
    plt.close()
    
def plot_potential_individual(galaxy, grid,
                                    units='arcsec', kpc_per_arcsec=None,
                                    xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                    linscale=0.01,
                                    figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                                    title='Galaxy Potential', titlesize=46, xlabelsize=36, ylabelsize=36,
                                    output_path=None, output_format='show',
                                    output_filename='galaxy_individual_potential'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize = plotter_tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)
    plt.figure(figsize=figsize)
    as_subplot = True

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i+1)

        profile_plotters.plot_potential(mass_profile, grid, as_subplot,
                                        units, kpc_per_arcsec, xyticksize, norm,
                                        norm_min, norm_max, linthresh, linscale, figsize, aspect, cmap,
                                        cb_ticksize, title, titlesize, xlabelsize, ylabelsize,
                                        output_path, output_format, output_filename)

    plotter_tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                       output_format=output_format)
    plt.close()
    
def plot_deflections_y_individual(galaxy, grid,
                              units='arcsec', kpc_per_arcsec=None,
                              xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                              linscale=0.01,
                              figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                              title='Galaxy Deflections (y)', titlesize=46, xlabelsize=36, ylabelsize=36,
                              output_path=None, output_format='show',
                              output_filename='galaxy_individual_deflections_y'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize = plotter_tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)
    plt.figure(figsize=figsize)
    as_subplot = True
    
    for i, mass_profile in enumerate(galaxy.mass_profiles):
    
        plt.subplot(rows, columns, i+1)
    
        profile_plotters.plot_deflections_y(mass_profile, grid, as_subplot,
                                            units, kpc_per_arcsec, xyticksize, norm,
                                            norm_min, norm_max, linthresh, linscale, figsize, aspect, cmap,
                                            cb_ticksize, title, titlesize, xlabelsize, ylabelsize,
                                            output_path, output_format, output_filename)
    
    plotter_tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                       output_format=output_format)
    plt.close()


def plot_deflections_x_individual(galaxy, grid,
                                  units='arcsec', kpc_per_arcsec=None,
                                  xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                  linscale=0.01,
                                  figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                                  title='Galaxy Deflections (x)', titlesize=46, xlabelsize=36, ylabelsize=36,
                                  output_path=None, output_format='show',
                                  output_filename='galaxy_individual_deflections_x'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize = plotter_tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)
    plt.figure(figsize=figsize)
    as_subplot = True

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.plot_deflections_x(mass_profile, grid, as_subplot,
                                            units, kpc_per_arcsec, xyticksize, norm,
                                            norm_min, norm_max, linthresh, linscale, figsize, aspect, cmap,
                                            cb_ticksize, title, titlesize, xlabelsize, ylabelsize,
                                            output_path, output_format, output_filename)

    plotter_tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                       output_format=output_format)
    plt.close()