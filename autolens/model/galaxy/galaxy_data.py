import numpy as np

from autolens.data.array import grids, mask as msk, scaled_array


class GalaxyData(scaled_array.ScaledSquarePixelArray):

    def __new__(cls, array, noise_map, mask, sub_grid_size=2):
        return np.array(mask.map_2d_array_to_masked_1d_array(array)).view(cls)

    def __init__(self, array, noise_map, mask, sub_grid_size=2):
        """ A galaxy-data is a collection of data components which are used to fit a galaxy to another galaxy. \
        This is where a component of a galaxy's light profiles (e.g. intensities) or mass profiles (e.g. surface \
        density, potential or deflection angles) are fitted to one another.

        This is primarily performed for automatic prior linking, as a means to efficiently link the priors of a galaxy \
        using one inferred parametrization of light or mass profiles to a new galaxy with a different parametrization \
        of light or mass profiles.

        This omits a number of the data components typically used when fitting an regular (e.g. the observed regular, PSF, \
        exposure time map), but still has a number of the other components (e.g. an effective noise-map, grids).

        Parameters
        ----------
        array : scaled_array.ScaledSquarePixelArray
            An array of the quantity of the galaxy that is being fitted (e.g. its intensities, surface density, etc.).
        noise_map : scaled_array.ScaledSquarePixelArray
            The noise-map used for computing the likelihood of each fit. This can be chosen arbritarily.
        mask: msk.Mask
            The 2D mask that is applied to regular data.
        sub_grid_size : int
            The size of the sub-grid used for computing the SubGrid (see imaging.mask.SubGrid).

        Attributes
        ----------
        noise_map_ : ndarray
            The masked 1D array of the noise-map
        grids : imaging.mask.DataGrids
            Grids of (y,x) Cartesian coordinates which map over the masked 1D data array's pixels (includes an \
            regular-grid, sub-grid, etc.)
        padded_grids : imaging.mask.DataGrids
            Grids of padded (y,x) Cartesian coordinates which map over the every data array's pixel in 1D and a \
            padded regioon to include edge's for accurate PSF convolution (includes an regular-grid, sub-grid, etc.)
        """

        super().__init__(array=array, pixel_scale=array.pixel_scale)

        self.array = array
        self.mask = mask
        self.noise_map = noise_map
        self.noise_map_ = mask.map_2d_array_to_masked_1d_array(array_2d=noise_map)
        self.sub_grid_size = sub_grid_size

        self.grids = grids.DataGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                 sub_grid_size=sub_grid_size,
                                                                                 psf_shape=(1, 1))

        self.padded_grids = grids.DataGrids.padded_grids_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=sub_grid_size, psf_shape=(1, 1))

    def __array_finalize__(self, obj):
        super(GalaxyData, self).__array_finalize__(obj)
        if isinstance(obj, GalaxyData):
            self.array = obj.array
            self.mask = obj.mask
            self.noise_map = obj.noise_map
            self.noise_map_ = obj.noise_map_
            self.sub_grid_size = obj.sub_grid_size
            self.grids = obj.grids
            self.padded_grids = obj.padded_grids

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        raise NotImplementedError()


class GalaxyDataIntensities(GalaxyData):
    """ A galaxy-data collection for fitting two galaxies based on their light-profile intensities."""

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        """Generic function so that this galaxy-data type computes the correct profile quantity for the fitting.

        In this case, the light profile intensities are computed."""
        intensities = galaxy.intensities_from_grid(grid=sub_grid)
        return sub_grid.sub_data_to_regular_data(sub_array=intensities)


class GalaxyDataSurfaceDensity(GalaxyData):
    """ A galaxy-data collection for fitting two galaxies based on their mass-profile surface densities."""

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        """Generic function so that this galaxy-data type computes the correct profile quantity for the fitting.

        In this case, the mass profile surface density is computed."""
        surface_density = galaxy.surface_density_from_grid(grid=sub_grid)
        return sub_grid.sub_data_to_regular_data(sub_array=surface_density)


class GalaxyDataPotential(GalaxyData):
    """ A galaxy-data collection for fitting two galaxies based on their mass-profile potentials."""

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        """Generic function so that this galaxy-data type computes the correct profile quantity for the fitting.

        In this case, the mass profile potential is computed."""
        potential = galaxy.potential_from_grid(grid=sub_grid)
        return sub_grid.sub_data_to_regular_data(sub_array=potential)


class GalaxyDataDeflectionsY(GalaxyData):
    """ A galaxy-data collection for fitting two galaxies based on their mass-profile (y) deflection angles."""

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        """Generic function so that this galaxy-data type computes the correct profile quantity for the fitting.

        In this case, the mass profile deflection angles are computed."""
        deflections = galaxy.deflections_from_grid(grid=sub_grid)
        return np.asarray([sub_grid.sub_data_to_regular_data(deflections[:, 0]),
                           sub_grid.sub_data_to_regular_data(deflections[:, 1])]).T


class GalaxyDataDeflectionsX(GalaxyData):
    """ A galaxy-data collection for fitting two galaxies based on their mass-profile (x) deflection angles."""

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        """Generic function so that this galaxy-data type computes the correct profile quantity for the fitting.

        In this case, the mass profile deflection angles are computed."""
        deflections = galaxy.deflections_from_grid(grid=sub_grid)
        return np.asarray([sub_grid.sub_data_to_regular_data(deflections[:, 0]),
                           sub_grid.sub_data_to_regular_data(deflections[:, 1])]).T
