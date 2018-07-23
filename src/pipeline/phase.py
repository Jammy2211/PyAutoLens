from src.analysis import galaxy_prior
from src.analysis import galaxy
from src.analysis import ray_tracing
from src.imaging import mask as msk
from src.analysis import fitting


class Phase(object):
    def __init__(self, optimizer, sub_grid_size=1, blurring_shape=None):
        self.optimizer = optimizer
        self.sub_grid_size = sub_grid_size
        self.blurring_shape = blurring_shape

    def run(self, **kwargs):
        return self.optimizer.fit(self.make_analysis(**kwargs))

    def make_analysis(self, masked_image, last_results=None):
        masked_image = self.customize_image(masked_image, last_results)
        self.pass_priors(last_results)

        analysis = self.__class__.Analysis(sub_grid_size=self.sub_grid_size, blurring_shape=self.blurring_shape,
                                           masked_image=masked_image, last_results=last_results)
        return analysis

    @property
    def constant(self):
        return self.optimizer.constant

    @property
    def variable(self):
        return self.optimizer.variable

    class Analysis(object):
        def __init__(self, **kwargs):
            self.last_results = kwargs["last_results"]
            self.masked_image = kwargs["masked_image"]
            self.sub_grid_size = kwargs["sub_grid_size"]
            self.blurring_shape = kwargs["blurring_shape"] \
                if kwargs["blurring_shape"] is not None else self.masked_image.psf.shape
            self.coords_collection = msk.CoordinateCollection.from_mask_subgrid_size_and_blurring_shape(
                self.masked_image.mask, self.sub_grid_size, self.blurring_shape)

        def fit(self, **kwargs):
            raise NotImplementedError()

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def customize_image(self, masked_image, last_result):
        return masked_image

    def pass_priors(self, last_results):
        pass


class SourceLensPhase(Phase):

    @property
    def lens_galaxy(self):
        if hasattr(self.optimizer.constant, "lens_galaxy"):
            return self.optimizer.constant.lens_galaxy
        elif hasattr(self.optimizer.variable, "lens_galaxy"):
            return self.optimizer.variable.lens_galaxy

    @lens_galaxy.setter
    def lens_galaxy(self, lens_galaxy):
        if isinstance(lens_galaxy, galaxy.Galaxy):
            self.optimizer.constant.lens_galaxy = lens_galaxy
        elif isinstance(lens_galaxy, galaxy_prior.GalaxyPrior):
            self.optimizer.variable.lens_galaxy = lens_galaxy

    @property
    def source_galaxy(self):
        if hasattr(self.optimizer.constant, "source_galaxy"):
            return self.optimizer.constant.source_galaxy
        elif hasattr(self.optimizer.variable, "source_galaxy"):
            return self.optimizer.variable.source_galaxy

    @source_galaxy.setter
    def source_galaxy(self, source_galaxy):
        if isinstance(source_galaxy, galaxy.Galaxy):
            self.optimizer.constant.source_galaxy = source_galaxy
        elif isinstance(source_galaxy, galaxy_prior.GalaxyPrior):
            self.optimizer.variable.source_galaxy = source_galaxy


class InitialSourceLensPhase(SourceLensPhase):
    class Analysis(Phase.Analysis):

        def fit(self, lens_galaxy=None, source_galaxy=None):
            tracer = ray_tracing.Tracer([lens_galaxy], [source_galaxy], self.coords_collection)
            fitter = fitting.Fitter(self.masked_image, tracer)
            return fitter.fit_data_with_profiles()
