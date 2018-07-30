from src.analysis import galaxy_prior
from src.analysis import galaxy as g
from src.analysis import ray_tracing
from src.imaging import mask as msk
from src.imaging import masked_image as mi
from src.imaging import image as img
from src.analysis import fitting
from src.autopipe import non_linear
from src.autopipe import model_mapper as mm
import numpy as np
import inspect


class Phase(object):
    def __init__(self, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=lambda image: msk.Mask.circular(image.shape, image.pixel_scale, 3)):
        """
        A phase in an analysis pipeline. Uses the set non_linear optimizer to try to fit models and images passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        self.optimizer = optimizer_class()
        self.sub_grid_size = sub_grid_size
        self.mask_function = mask_function

    def run(self, image, last_results=None):
        """
        Run this phase.

        Parameters
        ----------
        last_results: non_linear.Result | None
            An object describing the results of the last phase or None if no phase has been executed
        image: img.Image
            An image that has been masked

        Returns
        -------
        result: non_linear.Result
            A result object comprising the best fit model and other data.
        """
        analysis = self.make_analysis(image=image, last_results=last_results)
        result = self.optimizer.fit(analysis)
        return self.__class__.Result(result.constant, result.likelihood, result.variable,
                                     galaxy_images=analysis.galaxy_images_for_model(result.constant))

    def make_analysis(self, image, last_results=None):
        """
        Create an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
        change the behaviour of the phase.

        Parameters
        ----------
        image: im.Image
            An image that has been masked
        last_results: non_linear.Result
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """
        mask = self.mask_function(image)
        masked_image = mi.MaskedImage(image, mask)
        masked_image = self.customize_image(masked_image, last_results)
        self.pass_priors(last_results)
        coords_collection = msk.GridCollection.from_mask_sub_grid_size_and_blurring_shape(
            masked_image.mask, self.sub_grid_size, masked_image.psf.shape)

        analysis = self.__class__.Analysis(coordinate_collection=coords_collection,
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

    class Result(non_linear.Result):

        def __init__(self, constant, likelihood, variable, galaxy_images):
            """
            The result of a phase

            Parameters
            ----------

            galaxy_images: [ndarray]
                A collection of images created by each individual galaxy which taken together form the full model image
            """
            super(Phase.Result, self).__init__(constant, likelihood, variable)
            self.galaxy_images = galaxy_images

        @property
        def model_image(self):
            return np.sum(np.stack(self.galaxy_images), axis=0)

    class Analysis(object):
        def __init__(self, last_results,
                     masked_image,
                     coordinate_collection):
            """
            An analysis object

            Parameters
            ----------
            last_results: Result
                The result of an analysis
            masked_image: mi.MaskedImage
                An image that has been masked
            coordinate_collection: msk.GridCollection
                A collection of coordinates (grid, blurring and sub)
            """
            self.last_results = last_results
            self.masked_image = masked_image
            self.coordinate_collection = coordinate_collection

        def fit(self, **kwargs):
            """
            Determine the fitness of a particular model

            Parameters
            ----------
            kwargs: dict
                Dictionary of objects describing the model

            Returns
            -------
            fit: fitting.Fit
                How fit the model is and the model
            """
            raise NotImplementedError()

        @property
        def hyper_galaxies(self):
            return [galaxy.hyper_galaxy for galaxy in self.last_results.constant.instances_of(g.Galaxy) if
                    galaxy.hyper_galaxy is not None]

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
    lens_galaxy = phase_property("lens_galaxy")
    source_galaxy = phase_property("source_galaxy")

    def __init__(self, lens_galaxy=None, source_galaxy=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1):
        """
        A phase with a simple source/lens model

        Parameters
        ----------
        lens_galaxy: g.Galaxy
            A galaxy that acts as a gravitational lens
        source_galaxy: g.Galaxy
            A galaxy that is being lensed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        super().__init__(optimizer_class=optimizer_class, sub_grid_size=sub_grid_size)
        self.lens_galaxy = lens_galaxy
        self.source_galaxy = source_galaxy

    class Result(Phase.Result):
        @property
        def lens_galaxy_image(self):
            return self.galaxy_images[0]

        @property
        def source_galaxy_image(self):
            return self.galaxy_images[1]

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
            fit: Fit
                A fractional value indicating how well this model fit and the model image itself
            """
            tracer = ray_tracing.Tracer([lens_galaxy], [source_galaxy], self.coordinate_collection)
            fitter = fitting.Fitter(self.masked_image, tracer)

            if self.last_results is not None:
                return fitter.fit_data_with_profiles_hyper_galaxies(self.last_results.model_image,
                                                                    self.last_results.galaxy_images,
                                                                    self.hyper_galaxies)

            return fitter.fit_data_with_profiles()

        def galaxy_images_for_model(self, model):
            """
            Generate images of galaxies for a set model.

            Parameters
            ----------
            model: mm.ModelInstance
                A model instance

            Returns
            -------
            galaxy_images: [ndarray]
                A list of images of galaxy components
            """
            tracer = ray_tracing.Tracer([model.lens_galaxy], [model.source_galaxy], self.coordinate_collection)
            return tracer.image_plane.galaxy_images, tracer.source_plane.galaxy_images

# result.lens_galaxy_images, result.source_galaxy_images =
