from src.analysis import galaxy_prior
from src.analysis import ray_tracing
from src.imaging import mask as msk
from src.analysis import fitting
from src.autopipe import non_linear
import inspect


class Phase(object):
    def __init__(self, optimizer_class=non_linear.MultiNest, sub_grid_size=1, blurring_shape=None):
        self.optimizer = optimizer_class()
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


def phase_property(name):
    def fget(self):
        if hasattr(self.optimizer.constant, name):
            return getattr(self.optimizer.constant, name)
        elif hasattr(self.optimizer.variable, name):
            return getattr(self.optimizer.variable, name)

    def fset(self, value):
        if inspect.isclass(value) or isinstance(value, galaxy_prior.GalaxyPrior):
            setattr(self.optimizer.variable, name, value)
            try:
                delattr(self.optimizer.constant, name)
            except AttributeError:
                pass
        else:
            setattr(self.optimizer.constant, name, value)
            try:
                delattr(self.optimizer.variable, name)
            except AttributeError:
                pass

    return property(fget=fget, fset=fset, doc=name)


class SourceLensPhase(Phase):
    lens_galaxy = phase_property("lens_galaxy")
    source_galaxy = phase_property("source_galaxy")
    

class InitialSourceLensPhase(SourceLensPhase):
    class Analysis(Phase.Analysis):

        def fit(self, lens_galaxy=None, source_galaxy=None):
            tracer = ray_tracing.Tracer([lens_galaxy], [source_galaxy], self.coords_collection)
            fitter = fitting.Fitter(self.masked_image, tracer)
            return fitter.fit_data_with_profiles()
