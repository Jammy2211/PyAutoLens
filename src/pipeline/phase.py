from src.analysis import galaxy_prior
from src.analysis import galaxy as g
from src.analysis import ray_tracing
from src.imaging import mask as msk
from src.imaging import masked_image as mi
from src.analysis import fitting
from src.autopipe import non_linear
import inspect


class Phase(object):
    def __init__(self, optimizer_class=non_linear.MultiNest, sub_grid_size=1):
        """
        A phase in an analysis pipeline. Uses the set non_linear optimizer to try to fit models and images passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        blurring_shape: (int, int)
            The shape of the PSF. Will default to the shape of the image PSF if none.
        """
        self.optimizer = optimizer_class()
        self.sub_grid_size = sub_grid_size

    def run(self, **kwargs):
        """
        Run this phase.

        Parameters
        ----------
        kwargs
            Arguments

        Returns
        -------
        result: non_linear.Result
            A result object comprising the best fit model and other data.
        """
        return self.optimizer.fit(self.make_analysis(**kwargs))

    def make_analysis(self, masked_image, last_results=None):
        """
        Create an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
        change the behaviour of the phase.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An image that has been masked
        last_results: non_linear.Result
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """
        masked_image = self.customize_image(masked_image, last_results)
        self.pass_priors(last_results)

        analysis = self.__class__.Analysis(sub_grid_size=self.sub_grid_size,
                                           masked_image=masked_image, last_results=last_results)
        return analysis

    @property
    def constant(self):
        """
        Convenience method

        Returns
        -------
        ModelInstance
            A model instance comprising all the constant objects in this analysis
        """
        return self.optimizer.constant

    @property
    def variable(self):
        """
        Convenience method

        Returns
        -------
        ModelMapper
            A model mapper comprising all the variable (prior) objects in this analysis
        """
        return self.optimizer.variable

    class Analysis(object):
        def __init__(self, **kwargs):
            """
            An analysis object

            Parameters
            ----------
            kwargs:
                Dictionary of arguments used in this analysis
            """
            self.last_results = kwargs["last_results"]
            self.masked_image = kwargs["masked_image"]
            self.sub_grid_size = kwargs["sub_grid_size"]
            self.coords_collection = msk.CoordinateCollection.from_mask_subgrid_size_and_blurring_shape(
                self.masked_image.mask, self.sub_grid_size, self.masked_image.psf.shape)

        def fit(self, **kwargs):
            """
            Determine the fitness of a particular model

            Parameters
            ----------
            kwargs: dict
                Dictionary of objects describing the model

            Returns
            -------
            float: fitness
                How fit the model is
            """
            raise NotImplementedError()

        @property
        def model_image(self):
            return self.last_results.model_image

        @property
        def galaxy_images(self):
            return self.last_results.galaxy_images

        @property
        def hyper_galaxies(self):
            return self.last_results.hyper_galaxies

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def customize_image(self, masked_image, last_result):
        """
        Customize an image. e.g. removing lens light.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An image that has been masked
        last_result: non_linear.Result
            The result of the previous analysis

        Returns
        -------
        masked_image: mi.MaskedImage
            The modified image (not changed by default)
        """
        return masked_image

    def pass_priors(self, last_results):
        """
        Perform any prior or constant passing. This could involve setting model attributes equal to priors or constants
        from a previous phase.

        Parameters
        ----------
        last_results: non_linear.Result
            The result of the previous phase
        """
        pass


def phase_property(name):
    """
    Create a property that is tied to the non_linear instance determines whether to set itself as a constant or
    variable.

    Parameters
    ----------
    name: str
        The name of this variable

    Returns
    -------
    property: property
        A property that appears to be an attribute of the phase but is really an attribute of constant or variable.
    """

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
    """
    A phase with a simple source/lens model
    """
    lens_galaxy = phase_property("lens_galaxy")
    source_galaxy = phase_property("source_galaxy")

    class Analysis(Phase.Analysis):

        def fit(self, lens_galaxy=None, source_galaxy=None):
            """
            Determine the fit of a lens galaxy and source galaxy to the image in this analysis.

            Parameters
            ----------
            lens_galaxy: g.Galaxy
                The galaxy that acts as a gravitational lens
            source_galaxy: g.Galaxy
                The galaxy that produces the light that is being lensed

            Returns
            -------
            fit: float
                A fractional value indicating how well this model fit
            """
            tracer = ray_tracing.Tracer([lens_galaxy], [source_galaxy], self.coords_collection)
            fitter = fitting.Fitter(self.masked_image, tracer)

            if self.last_results is not None:
                return fitter.fit_data_with_profiles_hyper_galaxies(self.model_image,
                                                                    self.galaxy_images,
                                                                    self.hyper_galaxies)

            return fitter.fit_data_with_profiles()
