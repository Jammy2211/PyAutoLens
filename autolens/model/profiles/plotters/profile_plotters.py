import numpy as np

from autolens.data.array.plotters import array_plotters

def plot_intensities(light_profile, grid, mask=None, positions=None, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                     cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Intensities', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                     mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                     output_path=None, output_format='show', output_filename='intensities'):
    """Plot the intensities (e.g. the image) of a light profile, on a regular grid of (y,x) coordinates.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    light_profile : model.profiles.light_profiles.LightProfile
        The light profile whose intensities are plotted.
    grid : ndarray or data.array.grid_stacks.RegularGrid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    intensities = light_profile.intensities_from_grid(grid=grid)
    intensities = grid.scaled_array_from_array_1d(intensities)

    array_plotters.plot_array(array=intensities, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              grid_pointsize=grid_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_surface_density(mass_profile, grid, mask=None, positions=None, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                         cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                         title='Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                         mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                         output_path=None, output_format='show', output_filename='surface_density'):
    """Plot the surface density of a mass profile, on a regular grid of (y,x) coordinates.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose surface density is plotted.
    grid : ndarray or data.array.grid_stacks.RegularGrid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    surface_density = mass_profile.surface_density_from_grid(grid=grid)
    surface_density = grid.scaled_array_from_array_1d(surface_density)

    array_plotters.plot_array(array=surface_density, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              grid_pointsize=grid_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_potential(mass_profile, grid, mask=None, positions=None, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                   cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Potential', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                   mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                   output_path=None, output_format='show', output_filename='potential'):
    """Plot the potential of a mass profile, on a regular grid of (y,x) coordinates.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose potential is plotted.
    grid : ndarray or data.array.grid_stacks.RegularGrid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    potential = mass_profile.potential_from_grid(grid=grid)
    potential = grid.scaled_array_from_array_1d(potential)

    array_plotters.plot_array(array=potential, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              grid_pointsize=grid_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_deflections_y(mass_profile, grid, mask=None, positions=None, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                       cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Deflections (y)', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                       mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                       output_path=None, output_format='show', output_filename='deflections_y'):
    """Plot the y component of the deflection angles of a mass profile, on a regular grid of (y,x) coordinates.

    Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose y deflecton angles are plotted.
    grid : ndarray or data.array.grid_stacks.RegularGrid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    deflections = mass_profile.deflections_from_grid(grid)
    deflections_y = grid.scaled_array_from_array_1d(deflections[:,0])

    array_plotters.plot_array(array=deflections_y, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              grid_pointsize=grid_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_deflections_x(mass_profile, grid, mask=None, positions=None, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='equal',
                       cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Deflections (x)', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                       mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                       output_path=None, output_format='show', output_filename='deflections_x'):
    """Plot the x component of the deflection angles of a mass profile, on a regular grid of (y,x) coordinates.

     Set *autolens.data.array.plotters.array_plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
     mass_profile : model.profiles.mass_profiles.MassProfile
         The mass profile whose x deflecton angles are plotted.
     grid : ndarray or data.array.grid_stacks.RegularGrid
         The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
     """
    deflections = mass_profile.deflections_from_grid(grid)
    deflections_x = grid.scaled_array_from_array_1d(deflections[:, 1])

    array_plotters.plot_array(array=deflections_x, mask=mask, positions=positions, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
                              cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                              linthresh=linthresh, linscale=linscale,
                              cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              xyticksize=xyticksize,
                              mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                              grid_pointsize=grid_pointsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_1d_density_as_function_of_radius(mass_profile, minimum_radius=1.0e-4, maximum_radius=10.0, radii_bins=10):
    radii_bin_size = (maximum_radius - minimum_radius) / (radii_bins - 1)
    annuli_radii = np.linspace(start=minimum_radius - radii_bin_size / 2.0, stop=maximum_radius + radii_bin_size / 2.0,
                               num=radii_bins + 1)
    density_radii = np.linspace(start=minimum_radius, stop=maximum_radius, num=radii_bins)
    pass