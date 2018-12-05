import logging
import os

import numpy as np
from autofit import conf
from autofit.core import non_linear

from autolens import exc
from autolens.data.array import mask as msk
from autolens.data.imaging import image as im
from autolens.data.imaging.plotters import imaging_plotters
from autolens.lensing import lensing_fitting
from autolens.lensing import lensing_image as li
from autolens.lensing import ray_tracing
from autolens.lensing import sensitivity_fitting
from autolens.lensing.plotters import sensitivity_fitting_plotters, lensing_fitting_plotters
from autolens.model.galaxy import galaxy as g, galaxy_model as gm, galaxy_fitting, galaxy_data as gd
from autolens.model.galaxy.plotters import galaxy_fitting_plotters
from autolens.model.inversion import pixelizations as pix
from autolens.pipeline.phase_property import PhasePropertyCollection

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


def default_mask_function(image):
    return msk.Mask.circular(image.shape, image.pixel_scale, 3.0)


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


class HyperOnly(object):

    def hyper_run(self, image, previous_results=None, mask=None):
        raise NotImplementedError()


class AbstractPhase(object):

    def __init__(self, optimizer_class=non_linear.MultiNest, phase_name=None, auto_link_priors=False):
        """
        A phase in an lensing pipeline. Uses the set non_linear optimizer to try to fit_normal models and image
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        phase_name: str
            The name of this phase
        """
        self.optimizer = optimizer_class(name=phase_name)
        self.phase_name = phase_name
        self.auto_link_priors = auto_link_priors

    @property
    def constant(self):
        """
        Convenience method

        Returns
        -------
        ModelInstance
            A model instance comprising all the constant objects in this lensing
        """
        return self.optimizer.constant

    @property
    def variable(self):
        """
        Convenience method

        Returns
        -------
        ModelMapper
            A model mapper comprising all the variable (prior) objects in this lensing
        """
        return self.optimizer.variable

    @property
    def galaxy_model_tuples(self):
        """
        Returns
        -------
        galaxy_model_tuples: [(String, GalaxyModel)]
            A list of tuples containing galaxy model names and instances.
        """
        return [tup for tup in self.optimizer.variable.prior_model_tuples if
                isinstance(tup.prior_model, gm.GalaxyModel)]

    def match_instance_to_models(self, instance):
        """
        Matches named galaxies associated with the instance to named galaxies associated with this phase.

        Parameters
        ----------
        instance: ModelInstance
            An instance with named galaxy attributes.

        Returns
        -------
        tuples: [(String, Galaxy, GalaxyModel)]
            A list of tuples associating galaxy instances from the model instance object with galaxy models in this
            phase.
        """
        galaxy_dict = dict(instance.name_instance_tuples_for_class(g.Galaxy))
        return [(key, galaxy_dict[key], value) for key, value in self.galaxy_model_tuples if key in galaxy_dict]

    def fit_priors(self, instance, fitting_function):
        """
        Update the priors in this phase by fitting each galaxy model to a galaxy with the same name from a previous
        phase if such a galaxy exists.

        Parameters
        ----------
        instance: ModelInstance
            An object with named galaxy attributes
        fitting_function: (Galaxy, GalaxyModel) -> GalaxyModel
            A function that takes a galaxy and a galaxy model and returns a GalaxyModel produced by combining a best fit
            between the original galaxy and galaxy model with prior widths given by the configuration.
        """
        tuples = self.match_instance_to_models(instance)
        for t in tuples:
            name = t[0]
            galaxy = t[1]
            galaxy_model = t[2]
            new_galaxy_model = fitting_function(galaxy, galaxy_model)
            for phase_property_collection in self.phase_property_collections:
                if hasattr(phase_property_collection, name):
                    setattr(phase_property_collection, name, new_galaxy_model)

    def fit_priors_with_results(self, results, fitting_function):
        """
        Update the priors in this phase by fitting each galaxy model to a galaxy with the same name from a previous
        phase if such a galaxy exists.

        Results later in the list take precedence, with the last instance of any galaxies that share a name being kept.

        Parameters
        ----------
        results: [Results]
            A list of results from previous phases.
        fitting_function: (Galaxy, GalaxyModel) -> GalaxyModel
            A function that takes a galaxy and a galaxy model and returns a GalaxyModel produced by combining a best fit
            between the original galaxy and galaxy model with prior widths given by the configuration.
        """
        if results is not None and len(results) > 0:
            instances = [r.constant for r in results]
            instance = instances[0]
            for next_instance in instances[1:]:
                instance += next_instance
            self.fit_priors(instance, fitting_function)

    @property
    def phase_property_collections(self):
        """
        Returns
        -------
        phase_property_collections: [PhasePropertyCollection]
            A list of phase property collections associated with this phase. This is used in automated prior passing and
            should be overridden for any phase that contains its own PhasePropertyCollections.
        """
        return []

    @property
    def path(self):
        return self.optimizer.path

    @property
    def doc(self):
        if self.__doc__ is not None:
            return self.__doc__.replace("  ", "").replace("\n", " ")

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

    # noinspection PyAbstractClass
    class Analysis(non_linear.Analysis):

        def __init__(self, phase_name, previous_results=None):
            """
            An lensing object

            Parameters
            ----------
            phase_name: str
                The name of the phase to which this analysis belongs
            previous_results: ResultsCollection
                The results of all previous phases
            """

            self.previous_results = previous_results
            self.phase_name = phase_name
            self.phase_output_path = "{}/{}".format(conf.instance.output_path, self.phase_name)

            log_file = conf.instance.general.get('output', 'log_file', str)
            if not len(log_file.replace(" ", "")) == 0:
                log_path = "{}/{}".format(self.phase_output_path, log_file)
                logger.handlers = [logging.FileHandler(log_path)]
                logger.propagate = False

            self.position_threshold = conf.instance.general.get('positions', 'position_threshold', float)
            self.plot_count = 0
            self.output_image_path = "{}/image/".format(self.phase_output_path)
            make_path_if_does_not_exist(path=self.output_image_path)

        @property
        def last_results(self):
            if self.previous_results is not None:
                return self.previous_results.last

        def tracer_for_instance(self, instance):
            raise NotImplementedError()

        def padded_tracer_for_instance(self, instance):
            raise NotImplementedError()

        def fit_for_tracers(self, tracer, padded_tracer):
            raise NotImplementedError()

        def fast_likelihood_for_tracer(self, tracer):
            raise NotImplementedError()

    class Result(non_linear.Result):

        def __init__(self, constant, likelihood, variable):
            """
            The result of a phase
            """
            super(Phase.Result, self).__init__(constant, likelihood, variable)


class Phase(AbstractPhase):

    def run(self, image, previous_results=None, mask=None):
        raise NotImplementedError()


class PhasePositions(AbstractPhase):
    lens_galaxies = PhasePropertyCollection("lens_galaxies")

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies]

    def __init__(self, lens_galaxies=None, optimizer_class=non_linear.MultiNest, phase_name=None,
                 auto_link_priors=False):
        super().__init__(optimizer_class, phase_name, auto_link_priors=auto_link_priors)
        self.lens_galaxies = lens_galaxies

    def run(self, positions, pixel_scale, previous_results=None):
        """
        Run this phase.

        Parameters
        ----------
        pixel_scale
        positions
        previous_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed

        Returns
        -------
        result: non_linear.Result
            A result object comprising the best fit_normal model and other datas.
        """
        analysis = self.make_analysis(positions=positions, pixel_scale=pixel_scale, previous_results=previous_results)
        result = self.optimizer.fit(analysis)
        return self.__class__.Result(result.constant, result.likelihood, result.variable)

    def make_analysis(self, positions, pixel_scale, previous_results=None):
        """
        Create an lensing object. Also calls the prior passing and lensing_image modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        pixel_scale
        positions
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        lensing: Analysis
            An lensing object that the non-linear optimizer calls to determine the fit_normal of a set of values
        """
        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(positions=positions, pixel_scale=pixel_scale, phase_name=self.phase_name,
                                           previous_results=previous_results)
        return analysis

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, positions, pixel_scale, phase_name, previous_results=None):
            super().__init__(phase_name, previous_results)

            self.positions = list(map(lambda position_set: np.asarray(position_set), positions))
            self.pixel_scale = pixel_scale

        def fit(self, instance):
            """
            Determine the fit_normal of a lens galaxy and source galaxy to the lensing_image in this lensing.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit_normal: Fit
                A fractional value indicating how well this model fit_normal and the model lensing_image itself
            """
            tracer = self.tracer_for_instance(instance)
            fit = self.fit_for_tracer(tracer)
            return fit.likelihood

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=instance.lens_galaxies,
                                                                positions=self.positions)

        def fit_for_tracer(self, tracer):
            return lensing_fitting.PositionFit(positions=tracer.source_plane.positions, noise=self.pixel_scale)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens lensing for... \n\nLens Galaxy::\n{}\n\n".format(instance.lens_galaxies))


class PhaseImaging(Phase):

    def __init__(self, optimizer_class=non_linear.MultiNest, sub_grid_size=1, image_psf_shape=None,
                 pixelization_psf_shape=None, positions=None, mask_function=default_mask_function, phase_name=None,
                 auto_link_priors=False):

        """

        A phase in an lensing pipeline. Uses the set non_linear optimizer to try to fit_normal models and image
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(PhaseImaging, self).__init__(optimizer_class, phase_name, auto_link_priors=auto_link_priors)
        self.sub_grid_size = sub_grid_size
        self.image_psf_shape = image_psf_shape
        self.pixelization_psf_shape = pixelization_psf_shape
        if positions is not None:
            self.positions = list(map(lambda position_set: np.asarray(position_set), positions))
        else:
            self.positions = None
        self.mask_function = mask_function

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def modify_image(self, image, previous_results):
        """
        Customize an lensing_image. e.g. removing lens light.

        Parameters
        ----------
        image: img.Image
            An lensing_image that has been masked
        previous_results: ResultsCollection
            The result of the previous lensing

        Returns
        -------
        lensing_image: img.Image
            The modified lensing_image (not changed by default)
        """
        return image

    def run(self, image, previous_results=None, mask=None):
        """
        Run this phase.

        Parameters
        ----------
        mask: Mask
            The default mask passed in by the pipeline
        previous_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        image: img.Image
            An lensing_image that has been masked

        Returns
        -------
        result: non_linear.Result
            A result object comprising the best fit_normal model and other datas.
        """
        analysis = self.make_analysis(image=image, previous_results=previous_results, mask=mask)
        result = self.optimizer.fit(analysis)
        analysis.visualize(instance=result.constant, suffix=None, during_analysis=False)

        return self.__class__.Result(result.constant, result.likelihood, result.variable, analysis)

    def make_analysis(self, image, previous_results=None, mask=None):
        """
        Create an lensing object. Also calls the prior passing and lensing_image modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        mask: Mask
            The default mask passed in by the pipeline
        image: im.Image
            An lensing_image that has been masked
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        lensing: Analysis
            An lensing object that the non-linear optimizer calls to determine the fit_normal of a set of values
        """
        mask = mask or self.mask_function(image)
        image = self.modify_image(image, previous_results)
        lensing_image = li.LensingImage(image=image, mask=mask, sub_grid_size=self.sub_grid_size,
                                        image_psf_shape=self.image_psf_shape, positions=self.positions)
        self.pass_priors(previous_results)

        analysis = self.__class__.Analysis(lensing_image=lensing_image, phase_name=self.phase_name,
                                           previous_results=previous_results)
        return analysis

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, lensing_image, phase_name, previous_results=None):

            super(PhaseImaging.Analysis, self).__init__(phase_name, previous_results)

            self.lensing_image = lensing_image

        def fit(self, instance):
            """
            Determine the fit_normal of a lens galaxy and source galaxy to the lensing_image in this lensing.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit_normal: Fit
                A fractional value indicating how well this model fit_normal and the model lensing_image itself
            """
            self.check_positions_trace_within_threshold(instance)
            tracer = self.tracer_for_instance(instance)
            return self.fast_likelihood_for_tracer(tracer)

        def visualize(self, instance, suffix, during_analysis):
            self.plot_count += 1

            tracer = self.tracer_for_instance(instance)
            padded_tracer = self.padded_tracer_for_instance(instance)
            fit = self.fit_for_tracers(tracer=tracer, padded_tracer=padded_tracer)

            imaging_plotters.plot_image_subplot(image=self.lensing_image.image, mask=self.lensing_image.mask,
                                                positions=self.lensing_image.positions,
                                                output_path=self.output_image_path,
                                                output_format='png', ignore_config=False)

            imaging_plotters.plot_image_individual(image=self.lensing_image.image, mask=self.lensing_image.mask,
                                                   positions=self.lensing_image.positions,
                                                   output_path=self.output_image_path, output_format='png')

            lensing_fitting_plotters.plot_fitting_subplot(fit=fit, output_path=self.output_image_path,
                                                          output_format='png',
                                                          ignore_config=False)

            lensing_fitting_plotters.plot_fitting_individuals(fit=fit, output_path=self.output_image_path,
                                                              output_format='png')

            return fit

        def fast_likelihood_for_tracer(self, tracer):
            return lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(lensing_image=self.lensing_image,
                                                                                 tracer=tracer)

        def fit_for_tracers(self, tracer, padded_tracer):
            return lensing_fitting.fit_lensing_image_with_tracer(lensing_image=self.lensing_image, tracer=tracer,
                                                                 padded_tracer=padded_tracer)

        def check_positions_trace_within_threshold(self, instance):

            if self.lensing_image.positions is not None:

                tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=instance.lens_galaxies,
                                                                      positions=self.lensing_image.positions)
                fit = lensing_fitting.PositionFit(positions=tracer.source_plane.positions,
                                                  noise=self.lensing_image.image.pixel_scale)

                if not fit.maximum_separation_within_threshold(self.position_threshold):
                    return exc.RayTracingException

        def map_to_1d(self, data):
            """Convenience method"""
            return self.lensing_image.mask.map_2d_array_to_masked_1d_array(data)

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super(PhaseImaging.Result, self).__init__(constant, likelihood, variable)

            tracer = analysis.tracer_for_instance(constant)
            padded_tracer = analysis.padded_tracer_for_instance(constant)
            self.fit = analysis.fit_for_tracers(tracer=tracer, padded_tracer=padded_tracer)


class LensPlanePhase(PhaseImaging):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = PhasePropertyCollection("lens_galaxies")

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies]

    def __init__(self, lens_galaxies=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1, image_psf_shape=None,
                 mask_function=default_mask_function, phase_name="lens_only_phase", auto_link_priors=False):
        super(LensPlanePhase, self).__init__(optimizer_class=optimizer_class,
                                             sub_grid_size=sub_grid_size,
                                             image_psf_shape=image_psf_shape,
                                             mask_function=mask_function,
                                             phase_name=phase_name,
                                             auto_link_priors=auto_link_priors)
        self.lens_galaxies = lens_galaxies

    class Analysis(PhaseImaging.Analysis):

        def __init__(self, lensing_image, phase_name, previous_results=None):
            super(LensPlanePhase.Analysis, self).__init__(lensing_image, phase_name, previous_results)

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerImagePlane(lens_galaxies=instance.lens_galaxies,
                                                image_plane_grids=[self.lensing_image.grids])

        def padded_tracer_for_instance(self, instance):
            return ray_tracing.TracerImagePlane(lens_galaxies=instance.lens_galaxies,
                                                image_plane_grids=[self.lensing_image.padded_grids])

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens lensing for... \n\nLens Galaxy::\n{}\n\n".format(instance.lens_galaxies))

    class Result(PhaseImaging.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """

            super(LensPlanePhase.Result, self).__init__(constant, likelihood, variable, analysis)

            self.padded_model_image = self.fit.unmasked_model_profile_images
            self.lens_galaxy_padded_model_images = self.fit.unmasked_model_profile_images_of_galaxies
            self.lens_subtracted_padded_image = analysis.lensing_image.image - self.padded_model_image


class LensPlaneHyperPhase(LensPlanePhase):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = PhasePropertyCollection("lens_galaxies")

    def __init__(self, lens_galaxies=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1, image_psf_shape=None,
                 mask_function=default_mask_function, phase_name="lens_only_hyper_phase", auto_link_priors=False):
        super(LensPlaneHyperPhase, self).__init__(lens_galaxies=lens_galaxies, optimizer_class=optimizer_class,
                                                  image_psf_shape=image_psf_shape,
                                                  sub_grid_size=sub_grid_size, mask_function=mask_function,
                                                  phase_name=phase_name,
                                                  auto_link_priors=auto_link_priors)

    class Analysis(LensPlanePhase.Analysis):

        def __init__(self, lensing_image, phase_name, previous_results=None):
            super(LensPlanePhase.Analysis, self).__init__(lensing_image, phase_name, previous_results)
            self.hyper_model_image = self.map_to_1d(previous_results.last.unmasked_model_profile_images)
            self.hyper_galaxy_images = list(map(lambda galaxy_image: self.map_to_1d(galaxy_image),
                                                previous_results.last.lens_galaxy_padded_model_images))
            self.hyper_minimum_values = len(self.hyper_galaxy_images) * [0.0]

        # TODO : Can we make a HyperPhase class that overwrites these for all HyperPhases?

        def fast_likelihood_for_tracer(self, tracer):
            return lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(
                lensing_image=self.lensing_image,
                tracer=tracer)

        def fit_for_tracers(self, tracer, padded_tracer):
            return lensing_fitting.fit_lensing_image_with_tracer(lensing_image=self.lensing_image, tracer=tracer,
                                                                 padded_tracer=padded_tracer)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens lensing for... \n\nHyper Lens Galaxy::\n{}\n\n".format(instance.lens_galaxies))


class LensLightHyperOnlyPhase(LensPlaneHyperPhase, HyperOnly):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = PhasePropertyCollection("lens_galaxies")

    def __init__(self, optimizer_class=non_linear.MultiNest, sub_grid_size=1, image_psf_shape=None,
                 mask_function=default_mask_function, phase_name="lens_only_hyper_phase", hyper_index=None,
                 auto_link_priors=False):
        super(LensLightHyperOnlyPhase, self).__init__(lens_galaxies=[], optimizer_class=optimizer_class,
                                                      image_psf_shape=image_psf_shape,
                                                      sub_grid_size=sub_grid_size, mask_function=mask_function,
                                                      phase_name=phase_name,
                                                      auto_link_priors=auto_link_priors)

        self.hyper_index = hyper_index

    def hyper_run(self, image, previous_results=None, mask=None):
        class LensGalaxyHyperPhase(LensLightHyperOnlyPhase):

            # noinspection PyShadowingNames
            def pass_priors(self, previous_results):
                use_hyper_galaxy = len(previous_results[-1].constant.lens_galaxies) * [None]
                # noinspection PyTypeChecker
                use_hyper_galaxy[self.hyper_index] = g.HyperGalaxy

                self.lens_galaxies = list(map(lambda lens_galaxy, use_hyper:
                                              gm.GalaxyModel.from_galaxy(lens_galaxy, hyper_galaxy=use_hyper),
                                              previous_results.last.constant.lens_galaxies, use_hyper_galaxy))

        hyper_result = previous_results[-1]

        for i in range(len(previous_results[-1].constant.lens_galaxies)):
            phase = LensGalaxyHyperPhase(optimizer_class=non_linear.MultiNest, sub_grid_size=self.sub_grid_size,
                                         mask_function=self.mask_function,
                                         phase_name=self.phase_name + '/lens_gal_' + str(i), hyper_index=i)

            phase.optimizer.n_live_points = 20
            phase.optimizer.sampling_efficiency = 0.8
            result = phase.run(image, previous_results, mask)
            hyper_result.constant.lens_galaxies[i].hyper_galaxy = result.constant.lens_galaxies[i].hyper_galaxy

        return hyper_result

    def make_analysis(self, image, previous_results=None, mask=None):
        """
        Create an lensing object. Also calls the prior passing and lensing_image modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        mask: Mask
            The default mask passed in by the pipeline
        image: im.Image
            An lensing_image that has been masked
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        lensing: Analysis
            An lensing object that the non-linear optimizer calls to determine the fit_normal of a set of values
        """
        mask = mask or self.mask_function(image)
        image = self.modify_image(image, previous_results)
        lensing_image = li.LensingImage(image, mask, sub_grid_size=self.sub_grid_size)
        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(lensing_image=lensing_image, phase_name=self.phase_name,
                                           previous_results=previous_results, hyper_index=self.hyper_index)
        return analysis

    class Analysis(LensPlaneHyperPhase.Analysis):

        def __init__(self, lensing_image, phase_name, previous_results=None, hyper_index=None):
            super(LensPlaneHyperPhase.Analysis, self).__init__(lensing_image, phase_name, previous_results)

            self.hyper_model_image = self.map_to_1d(previous_results.last.unmasked_model_profile_images)
            self.hyper_galaxy_images = list(map(lambda galaxy_image: self.map_to_1d(galaxy_image),
                                                previous_results.last.lens_galaxy_padded_model_images))
            self.hyper_galaxy_images = [self.hyper_galaxy_images[hyper_index]]
            self.hyper_minimum_values = len(self.hyper_galaxy_images) * [0.0]


class LensSourcePlanePhase(PhaseImaging):
    """
    Fit a simple source and lens system.
    """

    lens_galaxies = PhasePropertyCollection("lens_galaxies")
    source_galaxies = PhasePropertyCollection("source_galaxies")

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies, self.source_galaxies]

    def __init__(self, lens_galaxies=None, source_galaxies=None, optimizer_class=non_linear.MultiNest,
                 sub_grid_size=1, image_psf_shape=None, mask_function=default_mask_function,
                 positions=None, phase_name="source_lens_phase", auto_link_priors=False):
        """
        A phase with a simple source/lens model

        Parameters
        ----------
        lens_galaxies : [g.Galaxy] | [gm.GalaxyModel]
            A galaxy that acts as a gravitational lens
        source_galaxies: [g.Galaxy] | [gm.GalaxyModel]
            A galaxy that is being lensed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(LensSourcePlanePhase, self).__init__(optimizer_class=optimizer_class,
                                                   sub_grid_size=sub_grid_size,
                                                   image_psf_shape=image_psf_shape,
                                                   mask_function=mask_function,
                                                   positions=positions,
                                                   phase_name=phase_name,
                                                   auto_link_priors=auto_link_priors)
        self.lens_galaxies = lens_galaxies or []
        self.source_galaxies = source_galaxies or []

    class Analysis(PhaseImaging.Analysis):

        def __init__(self, lensing_image, phase_name, previous_results=None):
            self.lensing_image = lensing_image
            super(PhaseImaging.Analysis, self).__init__(phase_name, previous_results)

        def tracer_for_instance(self, instance):
            image_plane_grids = pix.setup_image_plane_pixelization_grid_from_galaxies_and_grids(
                galaxies=instance.source_galaxies, grids=self.lensing_image.grids)

            return ray_tracing.TracerImageSourcePlanes(lens_galaxies=instance.lens_galaxies,
                                                       source_galaxies=instance.source_galaxies,
                                                       image_plane_grids=[image_plane_grids],
                                                       border=self.lensing_image.border)

        def padded_tracer_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(lens_galaxies=instance.lens_galaxies,
                                                       source_galaxies=instance.source_galaxies,
                                                       image_plane_grids=[self.lensing_image.padded_grids])

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens/source lensing for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                    instance.lens_galaxies, instance.source_galaxies))


class LensSourcePlaneHyperPhase(LensSourcePlanePhase):
    """
    Fit a simple source and lens system.
    """

    lens_galaxies = PhasePropertyCollection("lens_galaxies")
    source_galaxies = PhasePropertyCollection("source_galaxies")

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies, self.source_galaxies]

    def __init__(self, lens_galaxies=None, source_galaxies=None, optimizer_class=non_linear.MultiNest,
                 sub_grid_size=1, positions=None, image_psf_shape=None, mask_function=default_mask_function,
                 phase_name="source_lens_phase", auto_link_priors=False):
        """
        A phase with a simple source/lens model

        Parameters
        ----------
        lens_galaxies : [g.Galaxy] | [gm.GalaxyModel]
            A galaxy that acts as a gravitational lens
        source_galaxies: [g.Galaxy] | [gm.GalaxyModel]
            A galaxy that is being lensed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        super(LensSourcePlaneHyperPhase, self).__init__(lens_galaxies=lens_galaxies,
                                                        source_galaxies=source_galaxies,
                                                        optimizer_class=optimizer_class, sub_grid_size=sub_grid_size,
                                                        positions=positions, image_psf_shape=image_psf_shape,
                                                        mask_function=mask_function,
                                                        phase_name=phase_name,
                                                        auto_link_priors=auto_link_priors)
        self.lens_galaxies = lens_galaxies
        self.source_galaxies = source_galaxies

    class Analysis(LensSourcePlanePhase.Analysis):

        def __init__(self, lensing_image, phase_name, previous_results=None):
            super(LensSourcePlanePhase.Analysis, self).__init__(lensing_image, phase_name, previous_results)

            self.hyper_model_image = self.map_to_1d(previous_results.last.model_datas)
            self.hyper_galaxy_images = list(map(lambda galaxy_image: self.map_to_1d(galaxy_image),
                                                previous_results.last.source_galaxies_blurred_image_plane_images))
            self.hyper_minimum_values = len(self.hyper_galaxy_images) * [0.0]

        def fast_likelihood_for_tracer(self, tracer):
            return lensing_fitting.fast_likelihood_from_lensing_image_and_tracer(self.lensing_image, tracer)

        def fit_for_tracer(self, tracer):
            return lensing_fitting.fit_lensing_image_with_tracer(self.lensing_image, tracer,
                                                                 self.hyper_model_image)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens/source lensing for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                    instance.lens_galaxies, instance.source_galaxies))


class LensMassAndSourceProfileHyperOnlyPhase(LensSourcePlaneHyperPhase, HyperOnly):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = PhasePropertyCollection("lens_galaxies")
    source_galaxies = PhasePropertyCollection("source_galaxies")

    def __init__(self, optimizer_class=non_linear.MultiNest, sub_grid_size=1, image_psf_shape=None,
                 mask_function=default_mask_function, phase_name="source_and_len_hyper_phase", hyper_index=None,
                 auto_link_priors=False):
        super(LensMassAndSourceProfileHyperOnlyPhase, self).__init__(lens_galaxies=[], source_galaxies=[],
                                                                     optimizer_class=optimizer_class,
                                                                     sub_grid_size=sub_grid_size,
                                                                     image_psf_shape=image_psf_shape,
                                                                     mask_function=mask_function,
                                                                     phase_name=phase_name,
                                                                     auto_link_priors=auto_link_priors)
        self.hyper_index = hyper_index

    def hyper_run(self, image, previous_results=None, mask=None):
        class SourceGalaxyHyperPhase(LensMassAndSourceProfileHyperOnlyPhase):
            # noinspection PyShadowingNames
            def pass_priors(self, previous_results):
                use_hyper_galaxy = len(previous_results[-1].constant.source_galaxies) * [None]
                # noinspection PyTypeChecker
                use_hyper_galaxy[self.hyper_index] = g.HyperGalaxy

                self.lens_galaxies = previous_results[-1].variable.lens_galaxies
                self.lens_galaxies[0].sie = previous_results[0].constant.lens_galaxies[0].sie
                self.source_galaxies = list(map(lambda source_galaxy, use_hyper:
                                                gm.GalaxyModel.from_galaxy(source_galaxy, hyper_galaxy=use_hyper),
                                                previous_results.last.constant.source_galaxies, use_hyper_galaxy))

        overall_result = previous_results[-1]

        for i in range(len(previous_results[-1].constant.source_galaxies)):
            phase = SourceGalaxyHyperPhase(optimizer_class=non_linear.MultiNest,
                                           sub_grid_size=self.sub_grid_size, mask_function=self.mask_function,
                                           phase_name=self.phase_name + '/src_gal_' + str(i), hyper_index=i)

            phase.optimizer.n_live_points = 20
            phase.optimizer.sampling_efficiency = 0.8
            result = phase.run(image, previous_results, mask)
            overall_result.constant.source_galaxies[i].hyper_galaxy = result.constant.source_galaxies[i].hyper_galaxy

        return overall_result

    def make_analysis(self, image, previous_results=None, mask=None):
        """
        Create an lensing object. Also calls the prior passing and lensing_image modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        mask: Mask
            The default mask passed in by the pipeline
        image: im.Image
            An lensing_image that has been masked
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        lensing: Analysis
            An lensing object that the non-linear optimizer calls to determine the fit_normal of a set of values
        """
        mask = mask or self.mask_function(image)
        image = self.modify_image(image, previous_results)
        lensing_image = li.LensingImage(image, mask, sub_grid_size=self.sub_grid_size)
        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(lensing_image=lensing_image, phase_name=self.phase_name,
                                           previous_results=previous_results, hyper_index=self.hyper_index)
        return analysis

    class Analysis(LensSourcePlaneHyperPhase.Analysis):

        def __init__(self, lensing_image, phase_name, previous_results=None, hyper_index=None):
            super(LensSourcePlaneHyperPhase.Analysis, self).__init__(lensing_image, phase_name, previous_results)

            self.hyper_model_image = self.map_to_1d(previous_results.last.model_datas)
            self.hyper_galaxy_images = list(map(lambda galaxy_image: self.map_to_1d(galaxy_image),
                                                previous_results.last.source_galaxies_blurred_image_plane_images))
            self.hyper_galaxy_images = [self.hyper_galaxy_images[hyper_index]]
            self.hyper_minimum_values = len(self.hyper_galaxy_images) * [0.0]


class GalaxyFitPhase(AbstractPhase):
    galaxy = PhasePropertyCollection("galaxy")

    def __init__(self, galaxy_data_class, galaxy=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name=None):
        """
        A phase in an lensing pipeline. Uses the set non_linear optimizer to try to fit_normal models and image
        passed to it.

        Parameters
        ----------
        galaxy_data_class: class<gd.GalaxyData>
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(GalaxyFitPhase, self).__init__(optimizer_class, phase_name)
        self.galaxy_data_class = galaxy_data_class
        self.galaxy = galaxy
        self.sub_grid_size = sub_grid_size
        self.mask_function = mask_function

    def run(self, array, noise_map, previous_results=None, mask=None):
        """
        Run this phase.

        Parameters
        ----------
        mask: Mask
            The default mask passed in by the pipeline
        noise_map
        array
        previous_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed

        Returns
        -------
        result: non_linear.Result
            A result object comprising the best fit_normal model and other datas.
        """
        analysis = self.make_analysis(array=array, noise_map=noise_map, previous_results=previous_results, mask=mask)
        result = self.optimizer.fit(analysis)

        return self.__class__.Result(result.constant, result.likelihood, result.variable, analysis)

    def make_analysis(self, array, noise_map, previous_results=None, mask=None):
        """
        Create an lensing object. Also calls the prior passing and lensing_image modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        mask: Mask
            The default mask passed in by the pipeline
        array
        noise_map
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        lensing: Analysis
            An lensing object that the non-linear optimizer calls to determine the fit_normal of a set of values
        """
        mask = mask or self.mask_function(array)
        galaxy_datas = self.galaxy_data_class(array=array, noise_map=noise_map, mask=mask,
                                              sub_grid_size=self.sub_grid_size)
        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(galaxy_data=galaxy_datas, phase_name=self.phase_name,
                                           previous_results=previous_results)
        return analysis

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, galaxy_data, phase_name, previous_results=None):
            super(GalaxyFitPhase.Analysis, self).__init__(phase_name, previous_results)

            self.galaxy_data = galaxy_data

        def fit(self, instance):
            """
            Determine the fit_normal of a lens galaxy and source galaxy to the lensing_image in this lensing.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit_normal: Fit
                A fractional value indicating how well this model fit_normal and the model lensing_image itself
            """
            return self.fast_likelihood_for_instance(instance)

        def visualize(self, instance, suffix, during_analysis):
            self.plot_count += 1
            fit = self.fit_for_instance(instance)

            galaxy_fitting_plotters.plot_single_subplot(fit=fit, output_path=self.output_image_path,
                                                        output_format='png', ignore_config=False)

            return fit

        def fast_likelihood_for_instance(self, instance):
            return galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas=[self.galaxy_data],
                                                                              model_galaxy=instance.galaxy[0])

        def fit_for_instance(self, instance):
            return galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_datas=[self.galaxy_data],
                                                              model_galaxy=instance.galaxy[0])

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning galaxy fit_normal for... \n\nGalaxy::\n{}\n\n".format(instance.galaxy))

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super(GalaxyFitPhase.Result, self).__init__(constant, likelihood, variable)
            self.fit = analysis.fit_for_instance(instance=constant)


class GalaxyFitIntensitiesPhase(GalaxyFitPhase):
    def __init__(self, galaxy=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name=None):
        super().__init__(gd.GalaxyDataIntensities, galaxy, optimizer_class, sub_grid_size, mask_function, phase_name)


class GalaxyFitSurfaceDensityPhase(GalaxyFitPhase):
    def __init__(self, galaxy=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name=None):
        super().__init__(gd.GalaxyDataSurfaceDensity, galaxy, optimizer_class, sub_grid_size, mask_function, phase_name)


class GalaxyFitPotentialPhase(GalaxyFitPhase):
    def __init__(self, galaxy=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name=None):
        super().__init__(gd.GalaxyDataPotential, galaxy, optimizer_class, sub_grid_size, mask_function, phase_name)


class GalaxyFitDeflectionsPhase(AbstractPhase):
    galaxy = PhasePropertyCollection("galaxy")

    def __init__(self, galaxy=None, optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name=None):
        """
        A phase in an lensing pipeline. Uses the set non_linear optimizer to try to fit_normal models and image
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(GalaxyFitDeflectionsPhase, self).__init__(optimizer_class, phase_name)
        self.galaxy = galaxy
        self.sub_grid_size = sub_grid_size
        self.mask_function = mask_function

    def run(self, array_y, array_x, noise_map, previous_results=None, mask=None):
        """
        Run this phase.

        Parameters
        ----------
        noise_map
        array_x
        array_y
        mask: Mask
            The default mask passed in by the pipeline
        previous_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed

        Returns
        -------
        result: non_linear.Result
            A result object comprising the best fit_normal model and other datas.
        """
        analysis = self.make_analysis(array_y=array_y, array_x=array_x, noise_map=noise_map,
                                      previous_results=previous_results, mask=mask)
        result = self.optimizer.fit(analysis)
        analysis.visualize(instance=result.constant, suffix=None, during_analysis=False)

        return self.__class__.Result(result.constant, result.likelihood, result.variable, analysis)

    def make_analysis(self, array_y, array_x, noise_map, previous_results=None, mask=None):
        """
        Create an lensing object. Also calls the prior passing and lensing_image modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        noise_map
        array_x
        array_y
        mask: Mask
            The default mask passed in by the pipeline
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        lensing: Analysis
            An lensing object that the non-linear optimizer calls to determine the fit_normal of a set of values
        """
        mask = mask or self.mask_function(array_y)
        galaxy_data_y = gd.GalaxyDataDeflectionsY(array=array_y, noise_map=noise_map, mask=mask,
                                                  sub_grid_size=self.sub_grid_size)
        galaxy_data_x = gd.GalaxyDataDeflectionsX(array=array_x, noise_map=noise_map, mask=mask,
                                                  sub_grid_size=self.sub_grid_size)
        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(galaxy_data_y=galaxy_data_y, galaxy_data_x=galaxy_data_x,
                                           phase_name=self.phase_name, previous_results=previous_results)
        return analysis

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, galaxy_data_y, galaxy_data_x, phase_name, previous_results=None):
            super(GalaxyFitDeflectionsPhase.Analysis, self).__init__(phase_name, previous_results)

            self.galaxy_data_y = galaxy_data_y
            self.galaxy_data_x = galaxy_data_x

        def fit(self, instance):
            """
            Determine the fit_normal of a lens galaxy and source galaxy to the lensing_image in this lensing.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit_normal: Fit
                A fractional value indicating how well this model fit_normal and the model lensing_image itself
            """
            return self.fast_likelihood_for_instance(instance)

        def visualize(self, instance, suffix, during_analysis):
            fit = self.fit_for_instance(instance)

            galaxy_fitting_plotters.plot_deflections_subplot(fit=fit, output_path=self.output_image_path,
                                                             output_format='png', ignore_config=False)

            return fit

        def fast_likelihood_for_instance(self, instance):
            return galaxy_fitting.fast_likelihood_from_galaxy_data_and_galaxy(galaxy_datas=[self.galaxy_data_y,
                                                                                            self.galaxy_data_x],
                                                                              model_galaxy=instance.galaxy[0])

        def fit_for_instance(self, instance):
            return galaxy_fitting.fit_galaxy_data_with_galaxy(galaxy_datas=[self.galaxy_data_y, self.galaxy_data_x],
                                                              model_galaxy=instance.galaxy[0])

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning galaxy fit_normal for... \n\nGalaxy::\n{}\n\n".format(instance.galaxy))

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase
            """
            super(GalaxyFitDeflectionsPhase.Result, self).__init__(constant, likelihood, variable)
            self.fit = analysis.fit_for_instance(instance=constant)


class SensitivityPhase(PhaseImaging):
    lens_galaxies = PhasePropertyCollection("lens_galaxies")
    source_galaxies = PhasePropertyCollection("source_galaxies")
    sensitive_galaxies = PhasePropertyCollection("sensitive_galaxies")

    def __init__(self, lens_galaxies=None, source_galaxies=None, sensitive_galaxies=None,
                 optimizer_class=non_linear.MultiNest, sub_grid_size=1,
                 mask_function=default_mask_function, phase_name=None):
        """
        A phase in an lensing pipeline. Uses the set non_linear optimizer to try to fit_normal models and image
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(SensitivityPhase, self).__init__(optimizer_class=optimizer_class, sub_grid_size=sub_grid_size,
                                               mask_function=mask_function, phase_name=phase_name)

        self.lens_galaxies = lens_galaxies or []
        self.source_galaxies = source_galaxies or []
        self.sensitive_galaxies = sensitive_galaxies or []

    # noinspection PyAbstractClass
    class Analysis(PhaseImaging.Analysis):

        def __init__(self, lensing_image, phase_name, previous_results=None):
            self.sensitivity_image = lensing_image
            super(PhaseImaging.Analysis, self).__init__(phase_name, previous_results)

        def fit(self, instance):
            """
            Determine the fit_normal of a lens galaxy and source galaxy to the lensing_image in this lensing.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit_normal: Fit
                A fractional value indicating how well this model fit_normal and the model lensing_image itself
            """
            tracer_normal = self.tracer_normal_for_instance(instance)
            tracer_sensitive = self.tracer_sensitive_for_instance(instance)
            return self.fast_likelihood_for_tracers(tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)

        def visualize(self, instance, suffix, during_analysis):
            tracer_normal = self.tracer_normal_for_instance(instance)
            tracer_sensitive = self.tracer_sensitive_for_instance(instance)
            fit = self.fit_for_tracers(tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)

            imaging_plotters.plot_image_subplot(image=self.sensitivity_image.image, mask=self.lensing_image.mask,
                                                positions=self.lensing_image.positions,
                                                output_path=self.output_image_path, output_format='png',
                                                ignore_config=False)

            imaging_plotters.plot_image_individual(image=self.sensitivity_image.image, mask=self.lensing_image.mask,
                                                   positions=self.lensing_image.positions,
                                                   output_path=self.output_image_path,
                                                   output_format='png')

            sensitivity_fitting_plotters.plot_fitting_subplot(fit=fit, output_path=self.output_image_path,
                                                              output_format='png')

            return fit

        def tracer_normal_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(lens_galaxies=instance.lens_galaxies,
                                                       source_galaxies=instance.source_galaxies,
                                                       image_plane_grids=[self.sensitivity_image.grids],
                                                       border=self.sensitivity_image.border)

        def tracer_sensitive_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(
                lens_galaxies=instance.lens_galaxies + instance.sensitive_galaxies,
                source_galaxies=instance.source_galaxies,
                image_plane_grids=[self.sensitivity_image.grids],
                border=self.sensitivity_image.border)

        def fast_likelihood_for_tracers(self, tracer_normal, tracer_sensitive):
            return sensitivity_fitting.SensitivityProfileFit.fast_likelihood(
                sensitivity_images=[self.sensitivity_image],
                tracer_normal=tracer_normal,
                tracer_sensitive=tracer_sensitive)

        def fit_for_tracers(self, tracer_normal, tracer_sensitive):
            return sensitivity_fitting.SensitivityProfileFit(sensitivity_images=[self.sensitivity_image],
                                                             tracer_normal=tracer_normal,
                                                             tracer_sensitive=tracer_sensitive)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens/source lensing for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n Sensitive "
                "Galaxy\n{}\n\n "
                "".format(instance.lens_galaxies, instance.source_galaxies, instance.sensitive_galaxies))


def make_path_if_does_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
