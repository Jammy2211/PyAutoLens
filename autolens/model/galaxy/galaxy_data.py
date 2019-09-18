from autolens import exc
from autolens.array import grids, scaled_array
from autolens.array import mask as msk

from autolens.array.mapping import reshape_returned_array


class GalaxyData(object):
    def __init__(self, image, noise_map, pixel_scale):
        """ A galaxy-fit data_type is a collection of fit data_type components which are used to fit a galaxy to another galaxy. \
        This is where a component of a galaxy's light profiles (e.g. image) or mass profiles (e.g. convergence \
        , potential or deflection angles) are fitted to one another.

        This is primarily performed for automatic prior linking, as a means to efficiently link the priors of a galaxy \
        using one inferred parametrization of light or mass profiles to a new galaxy with a different parametrization \
        of light or mass profiles.

        This omits a number of the fit data_type components typically used when fitting an image (e.g. the observed image, PSF, \
        exposure time map), but still has a number of the other components (e.g. an effective noise_map-map, grid_stacks).

        Parameters
        ----------
        image : scaled_array.ScaledSquarePixelArray
            An image of the quantity of the galaxy that is being fitted (e.g. its image, convergence, etc.).
        noise_map : scaled_array.ScaledSquarePixelArray
            The noise_map-map used for computing the likelihood of each fit. This can be chosen arbritarily.
        """
        self.image = image
        self.noise_map = noise_map
        self.pixel_scale = pixel_scale


class GalaxyFitData(object):
    def __init__(
        self,
        galaxy_data,
        mask,
        pixel_scale_interpolation_grid=None,
        use_image=False,
        use_convergence=False,
        use_potential=False,
        use_deflections_y=False,
        use_deflections_x=False,
    ):
        """ A galaxy-fit data_type is a collection of fit data_type components which are used to fit a galaxy to another galaxy. \
        This is where a component of a galaxy's light profiles (e.g. image) or mass profiles (e.g. surface \
        density, potential or deflection angles) are fitted to one another.

        This is primarily performed for automatic prior linking, as a means to efficiently link the priors of a galaxy \
        using one inferred parametrization of light or mass profiles to a new galaxy with a different parametrization \
        of light or mass profiles.

        This omits a number of the fit data_type components typically used when fitting an image (e.g. the observed image, PSF, \
        exposure time map), but still has a number of the other components (e.g. an effective noise_map-map, grid_stacks).

        Parameters
        ----------
        galaxy_data : GalaxyData
            The collection of data_type about the galaxy (image of its profile map, noise-map, etc.) that is fitted.
        mask: msk.Mask
            The 2D masks that is applied to image fit data_type.
        sub_size : int
            The size of the sub-grid used for computing the SubGrid (see imaging.masks.SubGrid).

        Attributes
        ----------
        noise_map_1d : ndarray
            The masked 1D array of the noise_map-map
        grid_stacks : imaging.masks.GridStack
            Grids of (y,x) Cartesian coordinates which map over the masked 1D fit data_type array's pixels (includes an \
            grid, sub-grid, etc.)
        """
        self.galaxy_data = galaxy_data
        self.pixel_scale = galaxy_data.pixel_scale

        self._image_1d = mask.mapping.array_1d_from_array_2d(array_2d=galaxy_data.image)
        self._noise_map_1d = mask.mapping.array_1d_from_array_2d(
            array_2d=galaxy_data.noise_map
        )
        self.signal_to_noise_map_1d = self._image_1d / self._noise_map_1d
        self._mask_1d = mask.mapping.array_1d_from_array_2d(array_2d=mask)

        self.sub_size = mask.sub_size

        self.grid = grids.Grid.from_mask(mask=mask)

        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid

        if pixel_scale_interpolation_grid is not None:

            self.grid = self.grid.new_grid_with_interpolator(
                pixel_scale_interpolation_grid=pixel_scale_interpolation_grid
            )

        self.mask = mask

        if all(
            not element
            for element in [
                use_image,
                use_convergence,
                use_potential,
                use_deflections_y,
                use_deflections_x,
            ]
        ):
            raise exc.GalaxyException(
                "The galaxy fit data_type has not been supplied with a use_ method."
            )

        if (
            sum(
                [
                    use_image,
                    use_convergence,
                    use_potential,
                    use_deflections_y,
                    use_deflections_x,
                ]
            )
            > 1
        ):
            raise exc.GalaxyException(
                "The galaxy fit data_type has not been supplied with multiple use_ methods, only supply "
                "one."
            )

        self.use_image = use_image
        self.use_convergence = use_convergence
        self.use_potential = use_potential
        self.use_deflections_y = use_deflections_y
        self.use_deflections_x = use_deflections_x

    @property
    def mapping(self):
        return self.mask.mapping

    @reshape_returned_array
    def image(self, return_in_2d=True, return_masked=False):
        return self.galaxy_data.image

    @reshape_returned_array
    def noise_map(self, return_in_2d=True, return_masked=False):
        return self.galaxy_data.noise_map

    def __array_finalize__(self, obj):
        super(GalaxyFitData, self).__array_finalize__(obj)
        if isinstance(obj, GalaxyFitData):
            self.galaxy_data = obj.galaxy_data
            self.pixel_scale = obj.pixel_scale
            self.mask = obj.mask
            self._image_1d = obj._image_1d
            self._noise_map_1d = obj._noise_map_1d
            self._mask_1d = obj._mask_1d
            self.sub_size = obj.sub_size
            self.pixel_scale_interpolation_grid = obj.pixel_scale_interpolation_grid
            self.grid = obj.grid
            self.use_image = obj.use_image
            self.use_convergence = obj.use_convergence
            self.use_potential = obj.use_potential
            self.use_deflections_y = obj.use_deflections_y
            self.use_deflections_x = obj.use_deflections_x

    def profile_quantity_from_galaxies(self, galaxies):

        if self.use_image:
            return sum(
                map(
                    lambda g: g.profile_image_from_grid(
                        grid=self.grid, return_in_2d=False, return_binned=True
                    ),
                    galaxies,
                )
            )
        elif self.use_convergence:
            return sum(
                map(
                    lambda g: g.convergence_from_grid(
                        grid=self.grid, return_in_2d=False, return_binned=True
                    ),
                    galaxies,
                )
            )
        elif self.use_potential:
            return sum(
                map(
                    lambda g: g.potential_from_grid(
                        grid=self.grid, return_in_2d=False, return_binned=True
                    ),
                    galaxies,
                )
            )
        elif self.use_deflections_y:
            return sum(
                map(
                    lambda g: g.deflections_from_grid(
                        grid=self.grid, return_in_2d=False, return_binned=True
                    ),
                    galaxies,
                )
            )[:, 0]
        elif self.use_deflections_x:
            return sum(
                map(
                    lambda g: g.deflections_from_grid(
                        grid=self.grid, return_in_2d=False, return_binned=True
                    ),
                    galaxies,
                )
            )[:, 1]

    def mask(self, return_in_2d=True):
        if return_in_2d:
            return self.mask
        else:
            return self._mask_1d
