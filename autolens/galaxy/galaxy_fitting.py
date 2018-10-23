import numpy as np

from autolens.fitting import fitting


class GalaxyFit(fitting.AbstractDataFit):

    def __init__(self, galaxy_data, galaxy):


        super(GalaxyFit, self).__init__(fitting_data=galaxy_data, _model_data=_model_data)