import numpy as np

from autolens.fitting import fitting
from autolens.galaxy import galaxy_data as gd

def fit_galaxy_data_with_galaxy(galaxy_datas, galaxy):

    if isinstance(galaxy_datas[0], gd.GalaxyDataIntensities) or isinstance(galaxy_datas[0], gd.GalaxyDataSurfaceDensity) or \
        isinstance(galaxy_datas[0], gd.GalaxyDataPotential):
        return GalaxyFit(galaxy_datas=galaxy_datas, model_galaxy=galaxy)

    elif isinstance(galaxy_datas[0], gd.GalaxyDataDeflectionsY) or isinstance(galaxy_datas[0], gd.GalaxyDataDeflectionsX):
        return GalaxyFitDeflections(galaxy_datas=galaxy_datas, model_galaxy=galaxy)

def fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas, galaxy):

    if isinstance(galaxy_datas[0], gd.GalaxyDataIntensities) or isinstance(galaxy_datas[0], gd.GalaxyDataSurfaceDensity) or \
        isinstance(galaxy_datas[0], gd.GalaxyDataPotential):
        return GalaxyFit.fast_likelihood(galaxy_datas=galaxy_datas, galaxy=galaxy)

    elif isinstance(galaxy_datas[0], gd.GalaxyDataDeflectionsY) or isinstance(galaxy_datas[0], gd.GalaxyDataDeflectionsX):
        return GalaxyFitDeflections.fast_likelihood(galaxy_datas=galaxy_datas,  model_galaxy=galaxy)


class GalaxyFit(fitting.AbstractDataFit):

    def __init__(self, galaxy_datas, model_galaxy):

        self.galaxy_datas = galaxy_datas
        self.model_galaxy = model_galaxy

        _model_data = galaxy_datas[0].profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                            sub_grid=galaxy_datas[0].grids.sub)

        super(GalaxyFit, self).__init__(fitting_datas=galaxy_datas, _model_datas=[_model_data])

    @classmethod
    def fast_likelihood(cls, galaxy_datas, galaxy):
        _model_datas = galaxy_datas[0].profile_quantity_from_galaxy_and_sub_grid(galaxy=galaxy,
                                                                             sub_grid=galaxy_datas[0].grids.sub)
        _residuals = fitting.residuals_from_datas_and_model_datas([galaxy_datas[:]], [_model_datas])
        _chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals, [galaxy_datas[0].noise_map])
        chi_squared_terms = fitting.chi_squared_terms_from_chi_squareds(_chi_squareds)
        noise_terms = fitting.noise_terms_from_noise_maps([galaxy_datas[0].noise_map])
        return sum(fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms, noise_terms))


class GalaxyFitDeflections(fitting.AbstractDataFit):

    def __init__(self, galaxy_datas, model_galaxy):

        self.galaxy_datas = galaxy_datas
        self.model_galaxy = model_galaxy

        _model_data = galaxy_datas[0].profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                            sub_grid=galaxy_datas[0].grids.sub)

        super(GalaxyFitDeflections, self).__init__(fitting_datas=galaxy_datas, _model_datas=[_model_data[:, 0],
                                                                                              _model_data[:,1]])

    @classmethod
    def fast_likelihood(cls, galaxy_datas, model_galaxy):
        _model_datas = galaxy_datas[0].profile_quantity_from_galaxy_and_sub_grid(galaxy=model_galaxy,
                                                                              sub_grid=galaxy_datas[0].grids.sub)

        _residuals = fitting.residuals_from_datas_and_model_datas(galaxy_datas, [_model_datas[:, 0],
                                                                                 _model_datas[:,1]])
        _chi_squareds = fitting.chi_squareds_from_residuals_and_noise_maps(_residuals, galaxy_datas[0].noise_map)
        chi_squared_terms = fitting.chi_squared_terms_from_chi_squareds(_chi_squareds)
        noise_terms = fitting.noise_terms_from_noise_maps([galaxy_datas[0].noise_map, galaxy_datas[0].noise_map])
        return sum(fitting.likelihoods_from_chi_squareds_and_noise_terms(chi_squared_terms, noise_terms))