from src.analysis import galaxy_prior
from src.analysis import galaxy
from src.analysis import ray_tracing
from src.imaging import mask as msk
from src.analysis import fitting


class Phase(object):
    def __init__(self, optimizer, sub_grid_size=1, blurring_shape=None):
        self.optimizer = optimizer
        self.last_results = None
        self.masked_image = None
        self.coords_collection = None
        self.sub_grid_size = sub_grid_size
        self.__blurring_shape = blurring_shape

    @property
    def blurring_shape(self):
        return self.__blurring_shape \
            if self.__blurring_shape is not None or self.masked_image is None else self.masked_image.psf.shape

    @blurring_shape.setter
    def blurring_shape(self, blurring_shape):
        self.__blurring_shape = blurring_shape

    def run(self, **kwargs):
        self.last_results = kwargs["last_results"]
        self.masked_image = kwargs["masked_image"]
        self.coords_collection = msk.CoordinateCollection.from_mask_subgrid_size_and_blurring_shape(
            self.masked_image.mask, self.sub_grid_size, self.blurring_shape)


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

    def run(self, **kwargs):
        super(SourceLensPhase, self).run(**kwargs)

    def fit(self, **kwargs):
        tracer = ray_tracing.Tracer([kwargs["lens_galaxy"]], kwargs["source_galaxy"], self.coords_collection)
        fitter = fitting.Fitter(self.masked_image, tracer)
        return fitter.fit_data_with_profiles()
