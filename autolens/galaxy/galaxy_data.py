import numpy as np

from autolens.imaging import mask as msk
from autolens.imaging import scaled_array


class GalaxyData(scaled_array.ScaledSquarePixelArray):

    def __new__(cls, array, noise_map, mask, sub_grid_size=2):
        return np.array(mask.map_2d_array_to_masked_1d_array(array)).view(cls)

    def __init__(self, array, noise_map, mask, sub_grid_size=2):
        """
        The lensing datas_ is the collection of datas (image, noise-maps, PSF), a masks, grids, convolvers and other \
        utilities that are used for modeling and fitting an datas_ of a strong lens.

        Whilst the datas_ datas is initially loaded in 2D, for the lensing datas_ the masked-datas_ (and noise-maps) \
        are reduced to 1D arrays for faster calculations.

        Parameters
        ----------
        array : scaled_array.ScaledSquarePixelArray
            The original datas_ datas in 2D.
        mask: msk.Mask
            The 2D masks that is applied to the datas_.
        sub_grid_size : int
            The size of the sub-grid used for each lensing SubGrid. E.g. a value of 2 grids each datas_-pixel on a 2x2 \
            sub-grid.
        """

        super().__init__(array=array, pixel_scale=array.pixel_scale)

        self.array = array
        self.mask = mask
        self.noise_map = noise_map
        self.noise_map_ = mask.map_2d_array_to_masked_1d_array(array_2d=noise_map)
        self.sub_grid_size = sub_grid_size

        self.grids = msk.ImagingGrids.grids_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                                                  sub_grid_size=sub_grid_size,
                                                                                  psf_shape=(1, 1))

        self.unmasked_grids = msk.ImagingGrids.padded_grids_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=sub_grid_size, psf_shape=(1, 1))

    def __array_finalize__(self, obj):
        super(GalaxyData, self).__array_finalize__(obj)
        if isinstance(obj, GalaxyData):
            self.array = obj.array
            self.mask = obj.mask
            self.noise_map_ = obj.noise_map_
            self.sub_grid_size = obj.sub_grid_size
            self.grids = obj.grids
            self.unmasked_grids = obj.unmasked_grids

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        raise NotImplementedError


class GalaxyDataIntensities(GalaxyData):

    def __init__(self, array, noise_map, mask, sub_grid_size=2):
        super().__init__(array=array, noise_map=noise_map, mask=mask, sub_grid_size=sub_grid_size)

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        intensities = galaxy.intensities_from_grid(grid=sub_grid)
        return sub_grid.sub_data_to_image(sub_array=intensities)


class GalaxyDataSurfaceDensity(GalaxyData):

    def __init__(self, array, noise_map, mask, sub_grid_size=2):
        super().__init__(array=array, noise_map=noise_map, mask=mask, sub_grid_size=sub_grid_size)

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        surface_density = galaxy.surface_density_from_grid(grid=sub_grid)
        return sub_grid.sub_data_to_image(sub_array=surface_density)


class GalaxyDataPotential(GalaxyData):

    def __init__(self, array, noise_map, mask, sub_grid_size=2):
        super().__init__(array=array, noise_map=noise_map, mask=mask, sub_grid_size=sub_grid_size)

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        potential = galaxy.potential_from_grid(grid=sub_grid)
        return sub_grid.sub_data_to_image(sub_array=potential)


class GalaxyDataDeflectionsY(GalaxyData):

    def __init__(self, array, noise_map, mask, sub_grid_size=2):
        super().__init__(array=array, noise_map=noise_map, mask=mask, sub_grid_size=sub_grid_size)

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        deflections = galaxy.deflections_from_grid(grid=sub_grid)
        return np.asarray([sub_grid.sub_data_to_image(deflections[:, 0]),
                           sub_grid.sub_data_to_image(deflections[:, 1])]).T


class GalaxyDataDeflectionsX(GalaxyData):

    def __init__(self, array, noise_map, mask, sub_grid_size=2):
        super().__init__(array=array, noise_map=noise_map, mask=mask, sub_grid_size=sub_grid_size)

    def profile_quantity_from_galaxy_and_sub_grid(self, galaxy, sub_grid):
        deflections = galaxy.deflections_from_grid(grid=sub_grid)
        return np.asarray([sub_grid.sub_data_to_image(deflections[:, 0]),
                           sub_grid.sub_data_to_image(deflections[:, 1])]).T
