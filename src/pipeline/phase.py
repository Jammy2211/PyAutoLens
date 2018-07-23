from src.analysis import galaxy_prior
from src.analysis import galaxy


class Phase(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer


class SourceLensPhase(Phase):

    @property
    def lens_galaxy(self):
        return self.optimizer.constant.lens_galaxies + self.optimizer.variable.lens_galaxies

    @lens_galaxy.setter
    def lens_galaxy(self, lens_galaxy):
        if isinstance(lens_galaxy, galaxy.Galaxy):
            self.optimizer.constant.lens_galaxy = lens_galaxy
        elif isinstance(lens_galaxy, galaxy_prior.GalaxyPrior):
            self.optimizer.variable.lens_galaxy = lens_galaxy
