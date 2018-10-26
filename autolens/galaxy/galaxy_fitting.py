import numpy as np

from autolens.fitting import fitting
from autolens.galaxy import galaxy_data as gd

def fit_galaxy_data_with_galaxy(galaxy_data, galaxy):

    if isinstance(galaxy_data, gd.GalaxyDataIntensities) or isinstance(galaxy_data, gd.GalaxyDataSurfaceDensity) or \
        isinstance(galaxy_data, gd.GalaxyDataPotential):
        return GalaxyFit(galaxy_data=galaxy_data, model_galaxy=galaxy)

    elif isinstance(galaxy_data, gd.GalaxyDataDeflectionsY):
        return GalaxyFitDeflectionsY(galaxy_data=galaxy_data, model_galaxy=galaxy)

    elif isinstance(galaxy_data, gd.GalaxyDataDeflectionsX):
        return GalaxyFitDeflectionsX(galaxy_data=galaxy_data, model_galaxy=galaxy)

def fast_likelihood_from_galaxy_data_and_galaxy(galaxy_data, galaxy):

    if isinstance(galaxy_data, gd.GalaxyDataIntensities) or isinstance(galaxy_data, gd.GalaxyDataSurfaceDensity) or \
        isinstance(galaxy_data, gd.GalaxyDataPotential):
        return GalaxyFit.fast_likelihood(galaxy_data=galaxy_data, galaxy=galaxy)

    elif isinstance(galaxy_data, gd.GalaxyDataDeflectionsY):
        return GalaxyFitDeflectionsY.fast_likelihood(galaxy_data_y=galaxy_data, galaxy_data_x=galaxy_data,
                                                     model_galaxy=galaxy)

    elif isinstance(galaxy_data, gd.GalaxyDataDeflectionsX):
        return GalaxyFitDeflectionsX.fast_likelihood(galaxy_data_y=galaxy_data, galaxy_data_x=galaxy_data,
                                                     model_galaxy=galaxy)

class GalaxyFit(fitting.AbstractDataFit):

    def __init__(self, galaxy_data, model_galaxy):

        self.galaxy_data = galaxy_data
        self.model_galaxy = model_galaxy

        _model_data = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                            sub_grid=galaxy_data.grids.sub)

        super(GalaxyFit, self).__init__(fitting_datas=galaxy_data, _model_datas=_model_data)

    @classmethod
    def fast_likelihood(cls, galaxy_data, galaxy):
        _model_data = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                             sub_grid=galaxy_data.grids.sub)
        _residuals = fitting.residuals_from_datas_and_model_datas(galaxy_data[:], _model_data)
        _chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals, galaxy_data.noise_maps)
        chi_squared_term = fitting.chi_squared_terms_from_chi_squareds(_chi_squareds)
        noise_term = fitting.noise_terms_from_noise_maps(galaxy_data.noise_maps)
        return fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_term, noise_term)


class GalaxyFitDeflectionsY(fitting.AbstractDataFit):

    def __init__(self, galaxy_data, model_galaxy):

        self.galaxy_data = galaxy_data
        self.model_galaxy = model_galaxy

        _model_data = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                            sub_grid=galaxy_data.grids.sub)

        super(GalaxyFitDeflectionsY, self).__init__(fitting_datas=galaxy_data, _model_datas=_model_data[:, 0])

    @classmethod
    def fast_likelihood(cls, galaxy_data_y, galaxy_data_x, model_galaxy):
        _model_data = galaxy_data_y.profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                              sub_grid=galaxy_data_y.grids.sub)

        _residuals_y = fitting.residuals_from_datas_and_model_datas(galaxy_data_y[:], _model_data[:, 0])
        _chi_squareds_y = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals_y, galaxy_data_y.noise_maps)
        chi_squared_term_y = fitting.chi_squared_terms_from_chi_squareds(_chi_squareds_y)

        _residuals_x = fitting.residuals_from_datas_and_model_datas(galaxy_data_x[:], _model_data[:, 1])
        _chi_squareds_x = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals_x, galaxy_data_x.noise_maps)
        chi_squared_term_x = fitting.chi_squared_terms_from_chi_squareds(_chi_squareds_x)
        noise_term = 2.0*fitting.noise_terms_from_noise_maps(galaxy_data_y.noise_maps)
        return fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_term_y + chi_squared_term_x, noise_term)


class GalaxyFitDeflectionsX(fitting.AbstractDataFit):

    def __init__(self, galaxy_data, model_galaxy):

        self.galaxy_data = galaxy_data
        self.model_galaxy = model_galaxy

        _model_data = galaxy_data.profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                            sub_grid=galaxy_data.grids.sub)

        super(GalaxyFitDeflectionsX, self).__init__(fitting_datas=galaxy_data, _model_datas=_model_data[:, 1])

    @classmethod
    def fast_likelihood(cls, galaxy_data_y, galaxy_data_x, model_galaxy):
        _model_data = galaxy_data_y.profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                              sub_grid=galaxy_data_y.grids.sub)

        _residuals_y = fitting.residuals_from_datas_and_model_datas(galaxy_data_y[:], _model_data[:, 0])
        _chi_squareds_y = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals_y, galaxy_data_y.noise_maps)
        chi_squared_term_y = fitting.chi_squared_terms_from_chi_squareds(_chi_squareds_y)

        _residuals_x = fitting.residuals_from_datas_and_model_datas(galaxy_data_x[:], _model_data[:, 1])
        _chi_squareds_x = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals_x, galaxy_data_x.noise_maps)
        chi_squared_term_x = fitting.chi_squared_terms_from_chi_squareds(_chi_squareds_x)
        noise_term = 2.0*fitting.noise_terms_from_noise_maps(galaxy_data_y.noise_maps)
        return fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_term_y + chi_squared_term_x, noise_term)