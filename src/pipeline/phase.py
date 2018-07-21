from src.analysis import galaxy_prior
from src.analysis import galaxy


class Phase(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    @property
    def lens_galaxies(self):
        return self.optimizer.constant.lens_galaxies + self.optimizer.variable.lens_galaxies

    @lens_galaxies.setter
    def lens_galaxies(self, lens_galaxies):
        self.optimizer.constant.lens_galaxies = [lens_galaxy for lens_galaxy in lens_galaxies if
                                                 isinstance(lens_galaxy, galaxy.Galaxy)]
        self.optimizer.variable.lens_galaxies = [lens_galaxy for lens_galaxy in lens_galaxies if
                                                 isinstance(lens_galaxy, galaxy_prior.GalaxyPrior)]
