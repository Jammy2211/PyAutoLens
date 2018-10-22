import numpy as np

from autolens.fitting import fitting


class GalaxyFit(fitting.AbstractDataFit):

    def __init__(self, galaxy_data, galaxy):

        _model_data = galaxy.surface_density_from_grid(grid=galaxy_data.grids.sub)
        _model_data = galaxy_data.grids.sub.sub_data_to_image(sub_array=_model_data)

        super(GalaxyFit, self).__init__(fitting_data=galaxy_data, _model_data=_model_data)