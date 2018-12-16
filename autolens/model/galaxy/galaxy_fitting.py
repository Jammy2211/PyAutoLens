from autolens.data.fitting import fitter, fitting_util
from autolens.model.galaxy import galaxy_data as gd

def fit_galaxy_data_with_galaxy(galaxy_datas, model_galaxy):
    """Given a *galaxy_data.GalaxyData* object, fit it using a *galaxy.Galaxy* instance.

    This factory automatically pairs the two depending on the quantity being fitted.

    Parameters
    ----------
    galaxy_datas : [GalaxyData]
        The galaxy-datas objects being fitted.
    model_galaxy : galaxy.Galaxy
        The model galaxy used to fit the galaxy-datas.
    """
    if isinstance(galaxy_datas[0], gd.GalaxyDataIntensities) or isinstance(galaxy_datas[0], gd.GalaxyDataSurfaceDensity) or \
        isinstance(galaxy_datas[0], gd.GalaxyDataPotential):
        return GalaxyFit(galaxy_datas=galaxy_datas, model_galaxy=model_galaxy)

    elif isinstance(galaxy_datas[0], gd.GalaxyDataDeflectionsY) or isinstance(galaxy_datas[0], gd.GalaxyDataDeflectionsX):
        return GalaxyFitDeflections(galaxy_datas=galaxy_datas, model_galaxy=model_galaxy)

def fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas, model_galaxy):
    """Given a *galaxy_data.GalaxyData* object, fit it using a *model_galaxy.Galaxy* instance.

    The likelihood is computed in the fastest, least memory-intensive, way possible, for efficient non-linear sampling.

    This factory automatically pairs the two depending on the quantity being fitted.

    Parameters
    ----------
    galaxy_datas : [GalaxyData]
        The model_galaxy-datas objects being fitted.
    model_galaxy : model_galaxy.Galaxy
        The model model_galaxy used to fit the model_galaxy-datas.
    """
    if isinstance(galaxy_datas[0], gd.GalaxyDataIntensities) or isinstance(galaxy_datas[0], gd.GalaxyDataSurfaceDensity) or \
        isinstance(galaxy_datas[0], gd.GalaxyDataPotential):
        return GalaxyFit.fast_likelihood(galaxy_datas=galaxy_datas, galaxy=model_galaxy)

    elif isinstance(galaxy_datas[0], gd.GalaxyDataDeflectionsY) or isinstance(galaxy_datas[0], gd.GalaxyDataDeflectionsX):
        return GalaxyFitDeflections.fast_likelihood(galaxy_datas=galaxy_datas, model_galaxy=model_galaxy)


class GalaxyFit(fitter.DataFitterMulti):

    def __init__(self, galaxy_datas, model_galaxy):
        """Class which fits a set of galaxy-datas to a model galaxy, using either the galaxy's intensities, \
        surface-density or potential.

        Parameters
        ----------
        galaxy_datas : [GalaxyData]
            The galaxy-datas object being fitted.
        model_galaxy : galaxy.Galaxy
            The model galaxy used to fit the galaxy-datas.
        """

        self.galaxy_datas = galaxy_datas
        self.model_galaxy = model_galaxy

        model_data_ = galaxy_datas[0].profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                            sub_grid=galaxy_datas[0].grids.sub)

        super(GalaxyFit, self).__init__(fit_data=galaxy_datas, model_data=[model_data_])

    @classmethod
    def fast_likelihood(cls, galaxy_datas, galaxy):
        """Perform the fit of this class as described above, but storing no results as class instances, thereby \
        minimizing memory use and maximizing run-speed."""
        model_datas_ = galaxy_datas[0].profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                             sub_grid=galaxy_datas[0].grids.sub)
        residuals_ = fitting_util.residual_map_from_data_mask_and_model_data([galaxy_datas[:]], [model_datas_])
        chi_squareds_ = fitting_util.chi_squareds_from_residual_map_mask_and_noise_map(residuals_, [galaxy_datas[0].noise_map_])
        chi_squared_terms = fitting_util.chi_squared_term_from_chi_squared_map(chi_squareds_)
        noise_terms = fitting_util.noise_term_from_mask_and_noise_map([galaxy_datas[0].noise_map_])
        return sum(fitting_util.likelihood_from_chi_squared_term_and_noise_term(chi_squared_terms, noise_terms))


class GalaxyFitDeflections(fitter.DataFitterMulti):

    def __init__(self, galaxy_datas, model_galaxy):
        """Class which fits a set of galaxy-datas to a model galaxy, using the galaxy's deflection-angle maps.

        Parameters
        ----------
        galaxy_datas : [GalaxyData]
            The galaxy-datas object being fitted.
        model_galaxy : galaxy.Galaxy
            The model galaxy used to fit the galaxy-datas.
        """
        self.galaxy_datas = galaxy_datas
        self.model_galaxy = model_galaxy

        model_data_ = galaxy_datas[0].profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                            sub_grid=galaxy_datas[0].grids.sub)

        super(GalaxyFitDeflections, self).__init__(fit_data=galaxy_datas, model_data=[model_data_[:, 0],
                                                                                              model_data_[:,1]])

    @classmethod
    def fast_likelihood(cls, galaxy_datas, model_galaxy):
        """Perform the fit of this class as described above, but storing no results as class instances, thereby \
        minimizing memory use and maximizing run-speed."""
        model_datas_ = galaxy_datas[0].profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                              sub_grid=galaxy_datas[0].grids.sub)

        residuals_ = fitting_util.residual_map_from_data_mask_and_model_data(galaxy_datas, [model_datas_[:, 0],
                                                                                 model_datas_[:,1]])
        chi_squareds_ = fitting_util.chi_squareds_from_residual_map_mask_and_noise_map(residuals_, [galaxy_datas[0].noise_map_,
                                                                                                    galaxy_datas[0].noise_map_])
        chi_squared_terms = fitting_util.chi_squared_term_from_chi_squared_map(chi_squareds_)
        noise_terms = fitting_util.noise_term_from_mask_and_noise_map([galaxy_datas[0].noise_map_, galaxy_datas[0].noise_map_])
        return sum(fitting_util.likelihood_from_chi_squared_term_and_noise_term(chi_squared_terms, noise_terms))