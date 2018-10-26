from matplotlib import pyplot as plt

from autolens.plotting import tools
from autolens.plotting import tools_array
from autolens.plotting import profile_plotters

def plot_intensities(galaxy, grid, mask=None, positions=None, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                     cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Galaxy Intensities', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                     mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                     output_path=None, output_format='show', output_filename='galaxy_intensities'):

    intensities = galaxy.intensities_from_grid(grid=grid)
    intensities = grid.scaled_array_from_array_1d(intensities)

    tools_array.plot_array(array=intensities, mask=mask, positions=positions, as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                           grid_pointsize=grid_pointsize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_surface_density(galaxy, grid, mask=None, positions=None, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                         cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                         title='Galaxy Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                         mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                         output_path=None, output_format='show', output_filename='galaxy_surface_density'):

    surface_density = galaxy.surface_density_from_grid(grid=grid)
    surface_density = grid.scaled_array_from_array_1d(surface_density)

    tools_array.plot_array(array=surface_density, mask=mask, positions=positions, as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                           grid_pointsize=grid_pointsize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_potential(galaxy, grid, mask=None, positions=None, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                   cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Galaxy Potential', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                   mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                   output_path=None, output_format='show', output_filename='galaxy_potential'):

    potential = galaxy.potential_from_grid(grid=grid)
    potential = grid.scaled_array_from_array_1d(potential)

    tools_array.plot_array(array=potential, mask=mask, positions=positions, as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                           grid_pointsize=grid_pointsize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_deflections_y(galaxy, grid, mask=None, positions=None, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                       cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Galaxy Deflections (y)', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                       mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                       output_path=None, output_format='show', output_filename='galaxy_deflections_y'):

    deflections = galaxy.deflections_from_grid(grid)
    deflections_y = grid.scaled_array_from_array_1d(deflections[:,0])

    tools_array.plot_array(array=deflections_y, mask=mask, positions=positions, as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                           grid_pointsize=grid_pointsize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_deflections_x(galaxy, grid, mask=None, positions=None, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                       cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Galaxy Deflections (x)', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                       mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                       output_path=None, output_format='show', output_filename='galaxy_deflections_x'):

    deflections = galaxy.deflections_from_grid(grid)
    deflections_x = grid.scaled_array_from_array_1d(deflections[:,1])

    tools_array.plot_array(array=deflections_x, mask=mask, positions=positions, as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                           grid_pointsize=grid_pointsize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_intensities_subplot(galaxy, grid, mask=None, positions=None,
                             units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
                             cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                             cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                             titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                             mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                             output_path=None, output_format='show', output_filename='galaxy_individual_intensities'):

    total_light_profiles = len(galaxy.light_profiles)
    rows, columns, figsize_tool = tools.get_subplot_rows_columns_figsize(number_subplots=total_light_profiles)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for i, light_profile in enumerate(galaxy.light_profiles):

        plt.subplot(rows, columns, i+1)

        profile_plotters.plot_intensities(light_profile=light_profile, mask=mask, positions=positions, grid=grid,
                                          as_subplot=True,
                                          units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                          cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                                          linthresh=linthresh, linscale=linscale,
                                          cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                          title='Galaxy Component', titlesize=titlesize, xlabelsize=xlabelsize,
                                          ylabelsize=ylabelsize, xyticksize=xyticksize,
                                          mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                                          grid_pointsize=grid_pointsize,
                                          output_path=output_path, output_format=output_format,
                                          output_filename=output_filename)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                     output_format=output_format)
    plt.close()
    
def plot_surface_density_subplot(galaxy, grid, mask=None, positions=None,
                                 units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
                                 cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                                 linscale=0.01,
                                 cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                                 titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                                 mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                                 output_path=None, output_format='show',
                                 output_filename='galaxy_individual_surface_density'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i+1)

        profile_plotters.plot_surface_density(mass_profile=mass_profile, grid=grid, mask=mask, positions=positions,
                                              as_subplot=True,
                                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize,
                                              aspect=aspect,
                                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                                              linthresh=linthresh, linscale=linscale,
                                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                              title='Galaxy Component', titlesize=titlesize, xlabelsize=xlabelsize,
                                              ylabelsize=ylabelsize, xyticksize=xyticksize,
                                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                                              grid_pointsize=grid_pointsize,
                                              output_path=output_path, output_format=output_format,
                                              output_filename=output_filename)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                     output_format=output_format)
    plt.close()
    
def plot_potential_subplot(galaxy, grid, mask=None, positions=None,
                           units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
                           cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                           cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                           titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                           mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                           output_path=None, output_format='show',
                           output_filename='galaxy_individual_potential'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i+1)

        profile_plotters.plot_potential(mass_profile=mass_profile, grid=grid, mask=mask, positions=positions,
                                        as_subplot=True,
                                        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                        linscale=linscale,
                                        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                        title='Galaxy Component', titlesize=titlesize, xlabelsize=xlabelsize,
                                        ylabelsize=ylabelsize, xyticksize=xyticksize,
                                        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                                        grid_pointsize=grid_pointsize,
                                        output_path=output_path, output_format=output_format, output_filename=output_filename)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                     output_format=output_format)
    plt.close()
    
def plot_deflections_y_subplot(galaxy, grid, mask=None, positions=None,
                               units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
                               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                               linscale=0.01,
                               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                               titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                               mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                               output_path=None, output_format='show',
                               output_filename='galaxy_individual_deflections_y'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):
    
        plt.subplot(rows, columns, i+1)
    
        profile_plotters.plot_deflections_y(mass_profile=mass_profile, grid=grid, mask=mask, positions=positions,
                                            as_subplot=True,
                                            units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                                            linthresh=linthresh, linscale=linscale,
                                            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                            title='Galaxy Component', titlesize=titlesize, xlabelsize=xlabelsize,
                                            ylabelsize=ylabelsize, xyticksize=xyticksize,
                                            mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                                            grid_pointsize=grid_pointsize,
                                            output_path=output_path, output_format=output_format,
                                            output_filename=output_filename)
    
    tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                     output_format=output_format)
    plt.close()


def plot_deflections_x_subplot(galaxy, grid, mask=None, positions=None,
                               units='arcsec', kpc_per_arcsec=None, figsize=None, aspect='equal',
                               cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                               linscale=0.01,
                               cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                               titlesize=10, xlabelsize=10, ylabelsize=10, xyticksize=10,
                               mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                               output_path=None, output_format='show',
                               output_filename='galaxy_individual_deflections_x'):

    total_mass_profiles = len(galaxy.mass_profiles)
    rows, columns, figsize_tool = tools.get_subplot_rows_columns_figsize(number_subplots=total_mass_profiles)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        plt.subplot(rows, columns, i + 1)

        profile_plotters.plot_deflections_x(mass_profile=mass_profile, grid=grid, mask=mask, positions=positions,
                                            as_subplot=True,
                                            units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                                            cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                                            linthresh=linthresh, linscale=linscale,
                                            cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                                            title='Galaxy Component', titlesize=titlesize, xlabelsize=xlabelsize,
                                            ylabelsize=ylabelsize, xyticksize=xyticksize,
                                            mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                                            grid_pointsize=grid_pointsize,
                                            output_path=output_path, output_format=output_format,
                                            output_filename=output_filename)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                     output_format=output_format)
    plt.close()