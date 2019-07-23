import autofit as af

from autolens.data.array.grids import reshape_returned_regular_array


class GalaxyFit(af.DataFit1D):
    def __init__(self, galaxy_data, model_galaxies):
        """Class which fits a set of galaxy-datas to a model galaxy, using either the galaxy's intensities, \
        surface-density or potential.

        Parameters
        ----------
        galaxy_data : GalaxyData
            The galaxy-datas object being fitted.
        model_galaxies : galaxy.Galaxy
            The model galaxy used to fit the galaxy-datas.
        """

        self.galaxy_data = galaxy_data
        self.mask_2d = galaxy_data.mask_2d
        self.model_galaxies = model_galaxies
        self.map_to_scaled_array = galaxy_data.grid_stack.scaled_array_2d_from_array_1d

        model_data_1d = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(
            galaxies=model_galaxies, sub_grid=galaxy_data.grid_stack.sub
        )

        super(GalaxyFit, self).__init__(
            data_1d=galaxy_data.image_1d,
            noise_map_1d=galaxy_data.noise_map_1d,
            mask_1d=galaxy_data.mask_1d,
            model_data_1d=model_data_1d,
        )

    @property
    def grid_stack(self):
        return self.galaxy_data.grid_stack

    @property
    def image_1d(self):
        return self.data_1d

    @property
    def model_image_1d(self):
        return self.model_data_1d

    def mask(self, return_in_2d=True):
        if return_in_2d:
            return self.mask_2d
        else:
            return self.mask_1d

    @reshape_returned_regular_array
    def image(self):
        return self.image_1d

    @reshape_returned_regular_array
    def noise_map(self):
        return self.noise_map_1d

    @reshape_returned_regular_array
    def signal_to_noise_map(self):
        return self.signal_to_noise_map_1d

    @reshape_returned_regular_array
    def model_image(self):
        return self.model_image_1d

    @reshape_returned_regular_array
    def residual_map(self):
        return self.residual_map_1d

    @reshape_returned_regular_array
    def normalized_residual_map(self):
        return self.normalized_residual_map_1d

    @reshape_returned_regular_array
    def chi_squared_map(self):
        return self.chi_squared_map_1d

    @property
    def figure_of_merit(self):
        return self.likelihood
