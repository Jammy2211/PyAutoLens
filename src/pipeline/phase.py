from src.analysis import galaxy_prior as gp
from src.analysis import galaxy as g
from src.analysis import ray_tracing
from src.imaging import mask as msk
from src.imaging import masked_image as mi
from src.imaging import image as img
from src.analysis import fitting
from src.autopipe import non_linear
from src.autopipe import model_mapper as mm
import numpy as np
from src.pixelization import pixelization as px
import inspect
import logging

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


def default_mask_function(image):
    return msk.Mask.circular(image.shape, image.pixel_scale, 3)


class ResultsCollection(list):
    def __init__(self, results):
        super().__init__(results)

    @property
    def last(self):
        if len(self) > 0:
            return self[-1]
        return None

    @property
    def first(self):
        if len(self) > 0:
            return self[0]
        return None


class Phase(object):
    def __init__(self, optimizer_class=non_linear.DownhillSimplex, sub_grid_size=1,
                 mask_function=default_mask_function):
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
        last_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        image: img.Image
            An image that has been masked

        Returns
        -------
        result: non_linear.Result
            A result object comprising the best fit model and other data.
        """
        analysis = self.make_analysis(image=image, previous_results=last_results)
        result = self.optimizer.fit(analysis)
        return self.__class__.Result(result.constant, result.likelihood, result.variable,
                                     galaxy_images=analysis.galaxy_images_for_model(result.constant))

    def make_analysis(self, image, previous_results=None):
        """
        Create an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
        change the behaviour of the phase.

        Parameters
        ----------
        image: im.Image
            An image that has been masked
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """
        mask = self.mask_function(image)
        masked_image = mi.MaskedImage(image, mask)
        masked_image = self.customize_image(masked_image, previous_results)
        self.pass_priors(previous_results)
        coords_collection = msk.GridCollection.from_mask_sub_grid_size_and_blurring_shape(
            masked_image.mask, self.sub_grid_size, masked_image.psf.shape)

        analysis = self.__class__.Analysis(coordinate_collection=coords_collection,
                                           masked_image=masked_image, previous_results=previous_results)
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
        def __init__(self, previous_results,
                     masked_image,
                     coordinate_collection):
            """
            An analysis object

            Parameters
            ----------
            previous_results: ResultsCollection
                The results of all previous phases
            masked_image: mi.MaskedImage
                An image that has been masked
            coordinate_collection: msk.GridCollection
                A collection of coordinates (grid, blurring and sub)
            """
            self.previous_results = previous_results
            self.masked_image = masked_image
            self.coordinate_collection = coordinate_collection

        @property
        def last_results(self):
            if self.previous_results is not None:
                return self.previous_results.last

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

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def customize_image(self, masked_image, previous_results):
        """
        Customize an image. e.g. removing lens light.

        Parameters
        ----------
        masked_image: mi.MaskedImage
            An image that has been masked
        previous_results: ResultsCollection
            The result of the previous analysis

        Returns
        -------
        masked_image: mi.MaskedImage
            The modified image (not changed by default)
        """
        return masked_image

    def pass_priors(self, previous_results):
        """
        Perform any prior or constant passing. This could involve setting model attributes equal to priors or constants
        from a previous phase.

        Parameters
        ----------
        previous_results: ResultsCollection
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
        if inspect.isclass(value) or isinstance(value, gp.GalaxyPrior):
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

    def __init__(self,
                 lens_galaxy=None,
                 source_galaxy=None,
                 optimizer_class=non_linear.DownhillSimplex,
                 sub_grid_size=1,
                 mask_function=default_mask_function):
        """
        A phase with a simple source/lens model

        Parameters
        ----------
        lens_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that acts as a gravitational lens
        source_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that is being lensed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        super().__init__(optimizer_class=optimizer_class, sub_grid_size=sub_grid_size, mask_function=mask_function)
        self.lens_galaxy = lens_galaxy
        self.source_galaxy = source_galaxy

    def customize_image(self, masked_image, last_result):
        """

        Parameters
        ----------
        masked_image: mi.MaskedImage
        last_result: SourceLensPhase.Result

        Returns
        -------

        """
        return masked_image

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
            logger.debug(
                "\nRunning lens/source analysis for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                    lens_galaxy,
                    source_galaxy))

            tracer = ray_tracing.Tracer([lens_galaxy], [source_galaxy], self.coordinate_collection)
            fitter = fitting.Fitter(self.masked_image, tracer)

            if self.last_results is not None and lens_galaxy:
                return fitter.fit_data_with_profiles_and_hyper_galaxies(self.last_results.model_image,
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


class PixelizedSourceLensPhase(SourceLensPhase):
    def __init__(self,
                 lens_galaxy=None,
                 pixelization=px.RectangularRegConst,
                 optimizer_class=non_linear.DownhillSimplex,
                 sub_grid_size=1,
                 mask_function=default_mask_function):
        super().__init__(lens_galaxy=lens_galaxy, source_galaxy=gp.GalaxyPrior(pixelization=pixelization),
                         optimizer_class=optimizer_class, sub_grid_size=sub_grid_size, mask_function=mask_function)

    class Analysis(SourceLensPhase.Analysis):

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
            logger.debug(
                "\nRunning lens/source analysis for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                    lens_galaxy,
                    source_galaxy))

            tracer = ray_tracing.Tracer([lens_galaxy], [source_galaxy], self.coordinate_collection)
            fitter = fitting.PixelizedFitter(self.masked_image, "TODO", "TODO", tracer)

            if self.last_results is not None:
                return fitter.fit_data_with_pixelization_profiles_and_hyper_galaxies(self.last_results.model_image,
                                                                                     self.last_results.galaxy_images,
                                                                                     self.hyper_galaxies)

            return fitter.fit_data_with_pixelization_and_profiles()


class LensOnlyPhase(SourceLensPhase):
    def __init__(self,
                 lens_galaxy=None,
                 optimizer_class=non_linear.DownhillSimplex,
                 sub_grid_size=1,
                 mask_function=default_mask_function
                 ):
        super(LensOnlyPhase, self).__init__(lens_galaxy=lens_galaxy,
                                            optimizer_class=optimizer_class,
                                            sub_grid_size=sub_grid_size,
                                            mask_function=mask_function)


class SourceOnlyPhase(SourceLensPhase):
    def __init__(self,
                 source_galaxy=None,
                 optimizer_class=non_linear.DownhillSimplex,
                 sub_grid_size=1,
                 mask_function=default_mask_function
                 ):
        super(SourceOnlyPhase, self).__init__(source_galaxy=source_galaxy,
                                              optimizer_class=optimizer_class,
                                              sub_grid_size=sub_grid_size,
                                              mask_function=mask_function)


class SourceLensHyperGalaxyPhase(SourceLensPhase):
    def pass_priors(self, previous_results):
        self.lens_galaxy = gp.GalaxyPrior.from_galaxy(
            previous_results.last.constant.lens_galaxy,
            hyper_galaxy=g.HyperGalaxy)
        self.source_galaxy = gp.GalaxyPrior.from_galaxy(
            previous_results.last.constant.source_galaxy,
            hyper_galaxy=g.HyperGalaxy)
