from autofit.core import fitter

class GalaxyFit(fitter.DataFitter):

    def __init__(self, galaxy_data, model_galaxy):
        """Class which fits a set of galaxy-datas to a model galaxy, using either the galaxy's intensities, \
        surface-density or potential.

        Parameters
        ----------
        galaxy_data : GalaxyData
            The galaxy-datas object being fitted.
        model_galaxy : galaxy.Galaxy
            The model galaxy used to fit the galaxy-datas.
        """

        self.galaxy_data = galaxy_data
        self.model_galaxy = model_galaxy

        model_data_1d = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                              sub_grid=galaxy_data.grid_stack.sub)

        super(GalaxyFit, self).__init__(data=galaxy_data.array, noise_map=galaxy_data.noise_map,
                                        mask=galaxy_data.mask,
                                        model_data=galaxy_data.map_to_scaled_array(array_1d=model_data_1d))

    @property
    def figure_of_merit(self):
        return self.likelihood