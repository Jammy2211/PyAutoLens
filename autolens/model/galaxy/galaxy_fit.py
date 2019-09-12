import autofit as af

from autolens.array.mapping import reshape_returned_array


class GalaxyFit(af.DataFit):
    def __init__(self, galaxy_data, model_galaxies):
        """Class which fits a set of galaxy-datas to a model galaxy, using either the galaxy's image, \
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
        self.mapping = galaxy_data.mapping

        model_data_1d = galaxy_data.profile_quantity_from_galaxies(
            galaxies=model_galaxies
        )

        super(GalaxyFit, self).__init__(
            data=galaxy_data.image_1d,
            noise_map=galaxy_data.noise_map_1d,
            mask=galaxy_data.mask_1d,
            model_data=model_data_1d,
        )

    @property
    def grid(self):
        return self.galaxy_data.grid

    @reshape_returned_array
    def image(self, return_in_2d=True):
        return self._data

    @reshape_returned_array
    def noise_map(self, return_in_2d=True):
        return self._noise_map

    def mask(self, return_in_2d=True):
        if not return_in_2d:
            return self._mask
        else:
            return self.mapping.mask

    @reshape_returned_array
    def signal_to_noise_map(self, return_in_2d=True):
        return self._signal_to_noise_map

    @reshape_returned_array
    def model_image(self, return_in_2d=True):
        return self._model_data

    @reshape_returned_array
    def residual_map(self, return_in_2d=True):
        return self._residual_map

    @reshape_returned_array
    def normalized_residual_map(self, return_in_2d=True):
        return self._normalized_residual_map

    @reshape_returned_array
    def chi_squared_map(self, return_in_2d=True):
        return self._chi_squared_map

    @property
    def figure_of_merit(self):
        return self.likelihood
