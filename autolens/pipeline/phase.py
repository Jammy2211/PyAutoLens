from autolens.analysis import galaxy_prior as gp
from autolens.analysis import galaxy as g
from autolens.analysis import ray_tracing
from autolens.imaging import mask as msk
from autolens.imaging import masked_image as mi
from autolens.imaging import image as img
from autolens.analysis import fitting
from autolens.autopipe import non_linear
from autolens.autopipe import model_mapper as mm
from autolens import conf
import numpy as np
from autolens.pixelization import pixelization as px
import inspect
import logging
from astropy.io import fits

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


def default_mask_function(image):
    return msk.Mask.circular(image.shape_arc_seconds, image.pixel_scale, 3)


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


class IntervalCounter(object):
    def __init__(self, interval):
        self.count = 0
        self.interval = interval

    def __call__(self):
        self.count += 1
        return self.count % self.interval == 0


class Phase(object):
    def __init__(self, optimizer_class=non_linear.DownhillSimplex, sub_grid_size=1, mask_function=default_mask_function,
                 phase_name=None):
        """
        A phase in an analysis pipeline. Uses the set non_linear optimizer to try to fit models and images passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        self.optimizer = optimizer_class(name=phase_name)
        self.sub_grid_size = sub_grid_size
        self.mask_function = mask_function
        self.phase_name = phase_name

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

    @property
    def doc(self):
        if self.__doc__ is not None:
            return self.__doc__.replace("  ", "").replace("\n", " ")

    def run(self, image, last_results=None):
        """
        Run this phase.

        Parameters
        ----------
        last_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        image: img.Image
            An masked_image that has been masked

        Returns
        -------
        result: non_linear.Result
            A result object comprising the best fit model and other data.
        """
        analysis = self.make_analysis(image=image, previous_results=last_results)
        result = self.optimizer.fit(analysis)
        # self.visualise(analysis, result.constant)
        return self.__class__.Result(result.constant, result.likelihood, result.variable, analysis)

    def make_analysis(self, image, previous_results=None):
        """
        Create an analysis object. Also calls the prior passing and masked_image modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        image: im.Image
            An masked_image that has been masked
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """
        mask = self.mask_function(image)
        image = self.modify_image(image, previous_results)
        masked_image = mi.MaskedImage(image, mask, sub_grid_size=self.sub_grid_size)
        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(previous_results=previous_results, masked_image=masked_image,
                                           phase_name=self.phase_name)
        return analysis

    class Analysis(object):
        def __init__(self, previous_results, masked_image, phase_name):
            """
            An analysis object

            Parameters
            ----------
            previous_results: ResultsCollection
                The results of all previous phases
            masked_image: mi.MaskedImage
                An masked_image that has been masked
            """
            self.previous_results = previous_results
            self.phase_name = phase_name
            self.masked_image = masked_image
            log_interval = conf.instance.general.get('output', 'log_interval', int)
            self.__should_log = IntervalCounter(log_interval)
            visualise_interval = conf.instance.general.get('output', 'visualise_interval', int)
            self.__should_visualise = IntervalCounter(visualise_interval)

        @property
        def should_log(self):
            return self.__should_log()

        @property
        def should_visualise(self):
            return self.__should_visualise()

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

        def log(self, *args, **kwargs):
            raise NotImplementedError()

        def visualise(self, *args, **kwargs):
            raise NotImplementedError()

    class Result(non_linear.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super(Phase.Result, self).__init__(constant, likelihood, variable)
            self.galaxy_images = analysis.galaxy_images_for_model(constant)

        @property
        def model_image(self):
            return np.sum(np.stack((image for image in self.galaxy_images if image is not None)), axis=0)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def modify_image(self, image, previous_results):
        """
        Customize an masked_image. e.g. removing lens light.

        Parameters
        ----------
        image: img.Image
            An masked_image that has been masked
        previous_results: ResultsCollection
            The result of the previous analysis

        Returns
        -------
        masked_image: img.Image
            The modified masked_image (not changed by default)
        """
        return image

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
        The phase_name of this variable

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


class ProfileSourceLensPhase(Phase):
    """
    Fit a simple source and lens system.
    """

    lens_galaxy = phase_property("lens_galaxy")
    source_galaxy = phase_property("source_galaxy")

    def __init__(self, lens_galaxy=None, source_galaxy=None, optimizer_class=non_linear.DownhillSimplex,
                 sub_grid_size=1, mask_function=default_mask_function, phase_name="source_lens_phase"):
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
        super().__init__(optimizer_class=optimizer_class, sub_grid_size=sub_grid_size, mask_function=mask_function,
                         phase_name=phase_name)
        self.lens_galaxy = lens_galaxy
        self.source_galaxy = source_galaxy

    def modify_image(self, image, last_result):
        """

        Parameters
        ----------
        image: im.Image
        last_result: ProfileSourceLensPhase.Result

        Returns
        -------

        """
        return image

    class Analysis(Phase.Analysis):

        def __init__(self, masked_image, previous_results, phase_name):

            super(ProfileSourceLensPhase.Analysis, self).__init__(previous_results, masked_image, phase_name)

            self.hyper_model_image = None
            self.hyper_galaxy_images = None
            self.plot_count = 0
            if self.last_results is not None:
                self.hyper_model_image = self.masked_image.mask.map_to_1d(previous_results.last.model_image)
                self.hyper_galaxy_images = list(
                    map(self.masked_image.mask.map_to_1d, previous_results.last.galaxy_images))
                # TODO : We currently get these from tracer using defaults. Lets now set them up here via a config.
                # TODO : This is just a placehold for now
                self.hyper_minimum_values = len(self.hyper_galaxy_images) * [0.02]

        def fit(self, lens_galaxy=None, source_galaxy=None):
            """
            Determine the fit of a lens galaxy and source galaxy to the masked_image in this analysis.

            Parameters
            ----------
            lens_galaxy: g.Galaxy
                The galaxy that acts as a gravitational lens
            source_galaxy: g.Galaxy
                The galaxy that produces the light that is being lensed

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model masked_image itself
            """
            if self.should_log:
                self.log(lens_galaxy, source_galaxy)
            if self.should_visualise:
                self.visualise(lens_galaxy, source_galaxy)

            tracer = ray_tracing.Tracer(
                [] if lens_galaxy is None else [lens_galaxy],
                [] if source_galaxy is None else [source_galaxy], self.masked_image.grids)

            if self.last_results is None or not tracer.all_with_hyper_galaxies:

                fitter = fitting.ProfileFitter(self.masked_image, tracer)
                return fitter.blurred_image_likelihood

            elif self.last_results is not None and tracer.all_with_hyper_galaxies:

                fitter = fitting.HyperProfileFitter(self.masked_image, tracer, self.hyper_model_image,
                                                    self.hyper_galaxy_images, self.hyper_minimum_values)
                return fitter.blurred_image_scaled_likelihood

        @classmethod
        def log(cls, lens_galaxy, source_galaxy):
            logger.debug(
                "\nRunning lens/source analysis for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                    lens_galaxy, source_galaxy))

        def visualise(self, lens_galaxy, source_galaxy):
            self.plot_count += 1
            logger.info("Saving visualisations {}".format(self.plot_count))
            lens_image, source_image = self.galaxy_images_for_lens_galaxy_and_source_galaxy(lens_galaxy,
                                                                                            source_galaxy)

            def save_image(image, image_name):
                if image is not None:
                    hdu = fits.PrimaryHDU()
                    hdu.data = image
                    hdu.writeto("{}/{}/{}_{}.fits".format(conf.instance.data_path, self.phase_name, image_name,
                                                          self.plot_count))

            save_image(lens_image, "lens_image")
            save_image(source_image, "source_image")
            if lens_image is not None and source_image is not None:
                save_image(lens_image + source_image, "model_image")

        def galaxy_images_for_model(self, model):
            """
            Generate images of galaxies for a set model.

            Parameters
            ----------
            model: mm.ModelInstance
                A model instance

            Returns
            -------
            hyper_galaxy_images: [ndarray]
                A list of images of galaxy components
            """
            return self.galaxy_images_for_lens_galaxy_and_source_galaxy(model.lens_galaxy, model.source_galaxy)

        def galaxy_images_for_lens_galaxy_and_source_galaxy(self, lens_galaxy, source_galaxy):
            def model_image(plane):
                if len(plane.galaxies) == 0:
                    return None
                return self.masked_image.map_to_2d(plane.galaxy_images[0])

            lens_galaxies = [] if lens_galaxy is None else [lens_galaxy]
            source_galaxies = [] if source_galaxy is None else [source_galaxy]
            tracer = ray_tracing.Tracer(lens_galaxies, source_galaxies, self.masked_image.grids)
            return model_image(tracer.image_plane), model_image(tracer.source_plane)

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super(ProfileSourceLensPhase.Result, self).__init__(constant, likelihood, variable, analysis)

        @property
        def lens_galaxy_image(self):
            return self.galaxy_images[0]

        @property
        def source_galaxy_image(self):
            return self.galaxy_images[1]


class PixelizedSourceLensPhase(ProfileSourceLensPhase):
    """
    Fit a simple source and lens system using a pixelized source.
    """

    def __init__(self, lens_galaxy=None, pixelization=px.RectangularRegConst,
                 optimizer_class=non_linear.DownhillSimplex, sub_grid_size=1, mask_function=default_mask_function):
        super().__init__(lens_galaxy=lens_galaxy, source_galaxy=gp.GalaxyPrior(pixelization=pixelization),
                         optimizer_class=optimizer_class, sub_grid_size=sub_grid_size, mask_function=mask_function)

    class Analysis(ProfileSourceLensPhase.Analysis):

        def fit(self, lens_galaxy=None, source_galaxy=None):
            """
            Determine the fit of a lens galaxy and source galaxy to the masked_image in this analysis.

            Parameters
            ----------
            lens_galaxy: g.Galaxy
                The galaxy that acts as a gravitational lens
            source_galaxy: g.Galaxy
                The galaxy that produces the light that is being lensed

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model masked_image itself
            """
            if self.should_log:
                logger.debug(
                    "\nRunning lens/source analysis for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                        lens_galaxy,
                        source_galaxy))

            tracer = ray_tracing.Tracer([lens_galaxy], [source_galaxy], self.masked_image.grids)

            # TODO : Don't need a sparse mask for Rectangular pixelization. We'll remove it soon so just ignore it for
            # TODO : now.
            sparse_mask = None

            # TODO : I guess there is overhead doing this, and once we do it once in a fit we dont need to do it again.
            # TODO : Memoize or set up in class constructor?

            if tracer.has_galaxy_with_light_profile and not tracer.has_galaxy_with_pixelization:

                fitter = fitting.ProfileFitter(self.masked_image, tracer)
                return fitter.blurred_image_likelihood

            elif not tracer.has_galaxy_with_light_profile and tracer.has_galaxy_with_pixelization:

                fitter = fitting.PixelizationFitter(self.masked_image, sparse_mask, tracer)
                return fitter.reconstructed_image_evidence

            elif tracer.has_galaxy_with_light_profile and tracer.has_galaxy_with_pixelization:

                fitter = fitting.ProfileFitter(self.masked_image, tracer)
                pix_fitter = fitter.pixelization_fitter_with_profile_subtracted_masked_image(sparse_mask)
                return pix_fitter.reconstructed_image_evidence


class LensOnlyPhase(ProfileSourceLensPhase):
    """
    Fit only the lens galaxy light.
    """

    def __init__(self,
                 lens_galaxy=None,
                 optimizer_class=non_linear.DownhillSimplex,
                 sub_grid_size=1,
                 mask_function=default_mask_function,
                 phase_name="lens_only_phase"
                 ):
        super(LensOnlyPhase, self).__init__(lens_galaxy=lens_galaxy,
                                            optimizer_class=optimizer_class,
                                            sub_grid_size=sub_grid_size,
                                            mask_function=mask_function,
                                            phase_name=phase_name)


class SourceOnlyPhase(ProfileSourceLensPhase):
    """
    Fit only the source galaxy light and lens galaxy mass profile.
    """

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


class SourceLensHyperGalaxyPhase(ProfileSourceLensPhase):
    """
    Adjust hyper galaxy parameters to optimize the fit.
    """

    # TODO: Perform hyper galaxy analyses for each hyper galaxy independently
    def pass_priors(self, previous_results):
        self.lens_galaxy = gp.GalaxyPrior.from_galaxy(
            previous_results.last.constant.lens_galaxy,
            hyper_galaxy=g.HyperGalaxy)
        self.source_galaxy = gp.GalaxyPrior.from_galaxy(
            previous_results.last.constant.source_galaxy,
            hyper_galaxy=g.HyperGalaxy)
