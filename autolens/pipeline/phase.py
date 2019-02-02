import logging
import os
import warnings

import numpy as np
from astropy import cosmology as cosmo
from autofit import conf
from autofit.tools import phase
from autofit.tools.phase_property import PhasePropertyCollection
from autofit.optimize import non_linear

from autolens import exc
from autolens.data.array import mask as msk
from autolens.data.plotters import ccd_plotters
from autolens.lens import lens_data as li, lens_fit
from autolens.lens import ray_tracing
from autolens.lens import sensitivity_fit
from autolens.lens.plotters import sensitivity_fit_plotters, ray_tracing_plotters, lens_fit_plotters
from autolens.model.galaxy import galaxy as g, galaxy_model as gm, galaxy_fit, galaxy_data as gd
from autolens.model.galaxy.plotters import galaxy_fitting_plotters

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


def default_mask_function(image):
    return msk.Mask.circular(image.shape, image.pixel_scale, 3.0)

def setup_phase_mask(data, mask, mask_function, inner_circular_mask_radii):

    if mask_function is not None:
        mask = mask_function(image=data.image)
    elif mask is None and mask_function is None:
        mask = default_mask_function(image=data.image)

    if inner_circular_mask_radii is not None:
        inner_mask = msk.Mask.circular(shape=mask.shape, pixel_scale=mask.pixel_scale,
                                       radius_arcsec=inner_circular_mask_radii, invert=True)
        mask = mask + inner_mask

    return mask

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


class AbstractPhase(phase.AbstractPhase):

    def __init__(self, phase_name, optimizer_class=non_linear.MultiNest, cosmology=cosmo.Planck15,
                 auto_link_priors=False):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        phase_name: str
            The name of this phase
        """
        self.optimizer = optimizer_class(name=phase_name)
        self.cosmology = cosmology
        self.phase_name = phase_name
        self.auto_link_priors = auto_link_priors

    @property
    def constant(self):
        """
        Convenience method

        Returns
        -------
        ModelInstance
            A model instance comprising all the constant objects in this lens
        """
        return self.optimizer.constant

    @property
    def variable(self):
        """
        Convenience method

        Returns
        -------
        ModelMapper
            A model mapper comprising all the variable (prior) objects in this lens
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

        def __init__(self, cosmology, phase_name, previous_results=None):
            """
            An lens object

            Parameters
            ----------
            phase_name: str
                The name of the phase to which this analysis belongs
            previous_results: ResultsCollection
                The results of all previous phases
            """

            self.previous_results = previous_results
            self.cosmology = cosmology
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
            self.output_fits_path = "{}/image/fits/".format(self.phase_output_path)
            make_path_if_does_not_exist(path=self.output_fits_path)

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

        def figure_of_merit_for_fit(self, tracer):
            raise NotImplementedError()

    def make_result(self, result, analysis):
        return self.__class__.Result(constant=result.constant, figure_of_merit=result.figure_of_merit,
                                     variable=result.variable, analysis=analysis, optimizer=self.optimizer)

    class Result(non_linear.Result):

        def __init__(self, constant, figure_of_merit, variable, analysis, optimizer):
            """
            The result of a phase
            """
            super(Phase.Result, self).__init__(constant=constant, figure_of_merit=figure_of_merit, variable=variable)

            self.analysis = analysis
            self.optimizer = optimizer

        @property
        def most_likely_tracer(self):
            return self.analysis.tracer_for_instance(instance=self.constant)

        @property
        def most_likely_padded_tracer(self):
            return self.analysis.padded_tracer_for_instance(instance=self.constant)

        @property
        def most_likely_fit(self):
            return self.analysis.fit_for_tracers(tracer=self.most_likely_tracer,
                                                 padded_tracer=self.most_likely_padded_tracer)

        @property
        def unmasked_model_image(self):
            return self.most_likely_fit.unmasked_model_image

        @property
        def unmasked_model_image_of_planes(self):
            return self.most_likely_fit.unmasked_model_image_of_planes

        @property
        def unmasked_model_image_of_planes_and_galaxies(self):
            return self.most_likely_fit.unmasked_model_image_of_planes_and_galaxies


class Phase(AbstractPhase):

    def run(self, image, previous_results=None, mask=None):
        raise NotImplementedError()


class PhasePositions(AbstractPhase):
    lens_galaxies = PhasePropertyCollection("lens_galaxies")

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies]

    def __init__(self, phase_name, lens_galaxies=None, optimizer_class=non_linear.MultiNest, cosmology=cosmo.Planck15,
                 auto_link_priors=False):
        super().__init__(optimizer_class=optimizer_class, cosmology=cosmology,
                         phase_name=phase_name, auto_link_priors=auto_link_priors)
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
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper.
        """
        analysis = self.make_analysis(positions=positions, pixel_scale=pixel_scale, previous_results=previous_results)
        result = self.run_analysis(analysis)
        return self.make_result(result, analysis)

    def make_analysis(self, positions, pixel_scale, previous_results=None):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        pixel_scale
        positions
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens: Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """
        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(positions=positions, pixel_scale=pixel_scale, cosmology=self.cosmology,
                                           phase_name=self.phase_name, previous_results=previous_results)
        return analysis

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, positions, pixel_scale, cosmology, phase_name, previous_results=None):
            super().__init__(cosmology=cosmology, phase_name=phase_name, previous_results=previous_results)

            self.positions = list(map(lambda position_set: np.asarray(position_set), positions))
            self.pixel_scale = pixel_scale

        def visualize(self, instance, suffix, during_analysis):
            pass

        def fit(self, instance):
            """
            Determine the fit of a lens galaxy and source galaxy to the lens_data in this lens.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model lens_data itself
            """
            tracer = self.tracer_for_instance(instance)
            fit = self.fit_for_tracer(tracer)
            return fit.figure_of_merit

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=instance.lens_galaxies,
                                                                image_plane_positions=self.positions,
                                                                cosmology=self.cosmology)

        def fit_for_tracer(self, tracer):
            return lens_fit.LensPositionFit(positions=tracer.source_plane.positions, noise_map=self.pixel_scale)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens lens for... \n\nLens Galaxy::\n{}\n\n".format(instance.lens_galaxies))


class PhaseImaging(Phase):

    def __init__(self, phase_name, optimizer_class=non_linear.MultiNest, sub_grid_size=2, image_psf_shape=None,
                 pixelization_psf_shape=None, use_positions=False, mask_function=None, inner_circular_mask_radii=None,
                 cosmology=cosmo.Planck15, auto_link_priors=False):

        """

        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(PhaseImaging, self).__init__(optimizer_class=optimizer_class, cosmology=cosmology,
                                           phase_name=phase_name, auto_link_priors=auto_link_priors)
        self.sub_grid_size = sub_grid_size
        self.image_psf_shape = image_psf_shape
        self.pixelization_psf_shape = pixelization_psf_shape
        self.use_positions = use_positions
        self.mask_function = mask_function
        self.inner_circular_mask_radii = inner_circular_mask_radii

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def modify_image(self, image, previous_results):
        """
        Customize an lens_data. e.g. removing lens light.

        Parameters
        ----------
        image: scaled_array.ScaledSquarePixelArray
            An lens_data that has been masked
        previous_results: ResultsCollection
            The result of the previous lens

        Returns
        -------
        lens_data: scaled_array.ScaledSquarePixelArray
            The modified image (not changed by default)
        """
        return image

    def run(self, data, previous_results=None, mask=None, positions=None):
        """
        Run this phase.

        Parameters
        ----------
        mask: Mask
            The default masks passed in by the pipeline
        previous_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        data: scaled_array.ScaledSquarePixelArray
            An lens_data that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper.
        """
        analysis = self.make_analysis(data=data, previous_results=previous_results, mask=mask, positions=positions)

        result = self.run_analysis(analysis)

        return self.make_result(result, analysis)

    def make_analysis(self, data, previous_results=None, mask=None, positions=None):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        mask: Mask
            The default masks passed in by the pipeline
        data: im.CCD
            An lens_data that has been masked
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """

        mask = setup_phase_mask(data=data, mask=mask, mask_function=self.mask_function,
                                inner_circular_mask_radii=self.inner_circular_mask_radii)

        if self.use_positions and positions is not None:
            positions = list(map(lambda position_set: np.asarray(position_set), positions))
        elif not self.use_positions:
            positions = None
        elif self.use_positions and positions is None:
            raise exc.PhaseException('You have specified for a phase to use positions, but not input positions to the '
                                     'pipeline when you ran it.')

        lens_data = li.LensData(ccd_data=data, mask=mask, sub_grid_size=self.sub_grid_size,
                                image_psf_shape=self.image_psf_shape, positions=positions)

        modified_image = self.modify_image(image=lens_data.image, previous_results=previous_results)
        lens_data = lens_data.new_lens_data_with_modified_image(modified_image=modified_image)

        self.pass_priors(previous_results)

        self.output_phase_info()

        analysis = self.__class__.Analysis(lens_data=lens_data, cosmology=self.cosmology,
                                           phase_name=self.phase_name, previous_results=previous_results)
        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}/{}".format(conf.instance.output_path, self.phase_name, 'phase.info')

        with open(file_phase_info, 'w') as phase_info:

            phase_info.write('Optimizer = {} \n'.format(type(self.optimizer).__name__))
            phase_info.write('Sub-grid size = {} \n'.format(self.sub_grid_size))
            phase_info.write('Image PSF shape = {} \n'.format(self.image_psf_shape))
            phase_info.write('Pixelization PSF shape = {} \n'.format(self.pixelization_psf_shape))
            phase_info.write('Use positions = {} \n'.format(self.use_positions))
            position_threshold = conf.instance.general.get('positions', 'position_threshold', float)
            phase_info.write('Positions Threshold = {} \n'.format(position_threshold))
            phase_info.write('Cosmology = {} \n'.format(self.cosmology))
            phase_info.write('Auto Link Priors = {} \n'.format(self.auto_link_priors))

            phase_info.close()

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, lens_data, cosmology, phase_name, previous_results=None):

            super(PhaseImaging.Analysis, self).__init__(cosmology=cosmology, phase_name=phase_name,
                                                        previous_results=previous_results)

            self.lens_data = lens_data

        def fit(self, instance):
            """
            Determine the fit of a lens galaxy and source galaxy to the lens_data in this lens.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit : Fit
                A fractional value indicating how well this model fit and the model lens_data itself
            """
            self.check_positions_trace_within_threshold(instance)
            tracer = self.tracer_for_instance(instance)
            fit = self.fit_for_tracers(tracer=tracer, padded_tracer=None)
            return fit.figure_of_merit

        def visualize(self, instance, suffix, during_analysis):

            self.plot_count += 1

            tracer = self.tracer_for_instance(instance)
            padded_tracer = self.padded_tracer_for_instance(instance)
            fit = self.fit_for_tracers(tracer=tracer, padded_tracer=padded_tracer)

            ccd_plotters.plot_ccd_subplot(
                ccd_data=self.lens_data.ccd_data, mask=self.lens_data.mask, positions=self.lens_data.positions,
                output_path=self.output_image_path, output_format='png', ignore_config=False)

            ccd_plotters.plot_ccd_individual(
                ccd_data=self.lens_data.ccd_data, mask=self.lens_data.mask, positions=self.lens_data.positions,
                output_path=self.output_image_path, output_format='png')

            ray_tracing_plotters.plot_ray_tracing_subplot(
                tracer=tracer,
                output_path=self.output_image_path, output_format='png', ignore_config=False)

            lens_fit_plotters.plot_fit_subplot(
                fit=fit, positions=self.lens_data.positions, should_plot_image_plane_pix=True,
                output_path=self.output_image_path, output_format='png', ignore_config=False)

            if during_analysis:

                ray_tracing_plotters.plot_ray_tracing_individual(tracer=tracer, output_path=self.output_image_path,
                                                                 output_format='png', ignore_config=False)

                lens_fit_plotters.plot_fit_individuals(fit=fit, output_path=self.output_image_path,
                                                       output_format='png', ignore_config=False)

            elif not during_analysis:

                if conf.instance.general.get('output', 'plot_ray_tracing_all_at_end_png', bool):
                    ray_tracing_plotters.plot_ray_tracing_individual(tracer=tracer, output_path=self.output_image_path,
                                                                     output_format='png', ignore_config=True)

                if conf.instance.general.get('output', 'plot_ray_tracing_all_at_end_fits', bool):
                    ray_tracing_plotters.plot_ray_tracing_individual(tracer=tracer,
                                                                     output_path=self.output_fits_path,
                                                                     output_format='fits', ignore_config=True)

                if conf.instance.general.get('output', 'plot_lens_fit_all_at_end_png', bool):
                    lens_fit_plotters.plot_fit_individuals(fit=fit, output_path=self.output_image_path,
                                                           output_format='png', ignore_config=True)

                if conf.instance.general.get('output', 'plot_lens_fit_all_at_end_fits', bool):
                    lens_fit_plotters.plot_fit_individuals(fit=fit, output_path=self.output_fits_path,
                                                           output_format='fits', ignore_config=True)

            return fit

        def fit_for_tracers(self, tracer, padded_tracer):
            return lens_fit.fit_lens_data_with_tracer(lens_data=self.lens_data, tracer=tracer,
                                                      padded_tracer=padded_tracer)

        def check_positions_trace_within_threshold(self, instance):

            if self.lens_data.positions is not None:

                tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=instance.lens_galaxies,
                                                                      image_plane_positions=self.lens_data.positions)
                fit = lens_fit.LensPositionFit(positions=tracer.source_plane.positions,
                                               noise_map=self.lens_data.pixel_scale)

                if not fit.maximum_separation_within_threshold(self.position_threshold):
                    return exc.RayTracingException

        def map_to_1d(self, data):
            """Convenience method"""
            return self.lens_data.mask.map_2d_array_to_masked_1d_array(data)

    class Result(Phase.Result):

        def __init__(self, constant, figure_of_merit, variable, analysis, optimizer):
            """
            The result of a phase
            """
            super(PhaseImaging.Result, self).__init__(constant=constant, figure_of_merit=figure_of_merit,
                                                      variable=variable, analysis=analysis, optimizer=optimizer)


class LensPlanePhase(PhaseImaging):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = PhasePropertyCollection("lens_galaxies")

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies]

    def __init__(self, phase_name, lens_galaxies=None, optimizer_class=non_linear.MultiNest, sub_grid_size=2,
                 image_psf_shape=None, mask_function=None, inner_circular_mask_radii=None, cosmology=cosmo.Planck15,
                 auto_link_priors=False):
        super(LensPlanePhase, self).__init__(optimizer_class=optimizer_class,
                                             sub_grid_size=sub_grid_size,
                                             image_psf_shape=image_psf_shape,
                                             mask_function=mask_function,
                                             inner_circular_mask_radii=inner_circular_mask_radii,
                                             cosmology=cosmology,
                                             phase_name=phase_name,
                                             auto_link_priors=auto_link_priors)
        self.lens_galaxies = lens_galaxies

    class Analysis(PhaseImaging.Analysis):

        def __init__(self, lens_data, cosmology, phase_name, previous_results=None):
            super(LensPlanePhase.Analysis, self).__init__(lens_data=lens_data, cosmology=cosmology,
                                                          phase_name=phase_name, previous_results=previous_results)

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerImagePlane(lens_galaxies=instance.lens_galaxies,
                                                image_plane_grid_stack=self.lens_data.grid_stack,
                                                cosmology=self.cosmology)

        def padded_tracer_for_instance(self, instance):
            return ray_tracing.TracerImagePlane(lens_galaxies=instance.lens_galaxies,
                                                image_plane_grid_stack=self.lens_data.padded_grid_stack,
                                                cosmology=self.cosmology)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens lens for... \n\nLens Galaxy::\n{}\n\n".format(instance.lens_galaxies))

    class Result(PhaseImaging.Result):

        def __init__(self, constant, figure_of_merit, variable, analysis, optimizer):
            """
            The result of a phase
            """

            super(LensPlanePhase.Result, self).__init__(constant=constant, figure_of_merit=figure_of_merit,
                                                        variable=variable, analysis=analysis, optimizer=optimizer)

        @property
        def unmasked_lens_plane_model_image(self):
            return self.most_likely_fit.unmasked_model_image_of_planes[0]


class LensSourcePlanePhase(PhaseImaging):
    """
    Fit a simple source and lens system.
    """

    lens_galaxies = PhasePropertyCollection("lens_galaxies")
    source_galaxies = PhasePropertyCollection("source_galaxies")

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies, self.source_galaxies]

    def __init__(self, phase_name, lens_galaxies=None, source_galaxies=None, optimizer_class=non_linear.MultiNest,
                 sub_grid_size=2, image_psf_shape=None, use_positions=False, mask_function=None,
                 inner_circular_mask_radii=None, cosmology=cosmo.Planck15, auto_link_priors=False):
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
                                                   use_positions=use_positions,
                                                   mask_function=mask_function,
                                                   inner_circular_mask_radii=inner_circular_mask_radii,
                                                   cosmology=cosmology,
                                                   phase_name=phase_name,
                                                   auto_link_priors=auto_link_priors)
        self.lens_galaxies = lens_galaxies or []
        self.source_galaxies = source_galaxies or []

    class Analysis(PhaseImaging.Analysis):

        def __init__(self, lens_data, cosmology, phase_name, previous_results=None):
            self.lens_data = lens_data
            super(PhaseImaging.Analysis, self).__init__(cosmology=cosmology, phase_name=phase_name,
                                                        previous_results=previous_results)

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(lens_galaxies=instance.lens_galaxies,
                                                       source_galaxies=instance.source_galaxies,
                                                       image_plane_grid_stack=self.lens_data.grid_stack,
                                                       border=self.lens_data.border, cosmology=self.cosmology)

        def padded_tracer_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(lens_galaxies=instance.lens_galaxies,
                                                       source_galaxies=instance.source_galaxies,
                                                       image_plane_grid_stack=self.lens_data.padded_grid_stack,
                                                       cosmology=self.cosmology)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens/source lens for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                    instance.lens_galaxies, instance.source_galaxies))

    class Result(PhaseImaging.Result):

        def __init__(self, constant, figure_of_merit, variable, analysis, optimizer):
            """
            The result of a phase
            """

            super(LensSourcePlanePhase.Result, self).__init__(constant=constant, figure_of_merit=figure_of_merit,
                                                              variable=variable, analysis=analysis, optimizer=optimizer)

        @property
        def unmasked_lens_plane_model_image(self):
            return self.most_likely_fit.unmasked_model_image_of_planes[0]

        @property
        def unmasked_source_plane_model_image(self):
            return self.most_likely_fit.unmasked_model_image_of_planes[1]


class MultiPlanePhase(PhaseImaging):
    """
    Fit a simple source and lens system.
    """

    galaxies = PhasePropertyCollection("galaxies")

    @property
    def phase_property_collections(self):
        return [self.galaxies]

    def __init__(self, phase_name, galaxies=None, optimizer_class=non_linear.MultiNest,
                 sub_grid_size=2, image_psf_shape=None, use_positions=False, mask_function=None,
                 inner_circular_mask_radii=None, cosmology=cosmo.Planck15, auto_link_priors=False):
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

        super(MultiPlanePhase, self).__init__(optimizer_class=optimizer_class,
                                              sub_grid_size=sub_grid_size,
                                              image_psf_shape=image_psf_shape,
                                              use_positions=use_positions,
                                              mask_function=mask_function,
                                              inner_circular_mask_radii=inner_circular_mask_radii,
                                              cosmology=cosmology,
                                              phase_name=phase_name,
                                              auto_link_priors=auto_link_priors)
        self.galaxies = galaxies

    class Analysis(PhaseImaging.Analysis):

        def __init__(self, lens_data, cosmology, phase_name, previous_results=None):
            self.lens_data = lens_data
            super(PhaseImaging.Analysis, self).__init__(cosmology=cosmology, phase_name=phase_name,
                                                        previous_results=previous_results)

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerMultiPlanes(galaxies=instance.galaxies,
                                                 image_plane_grid_stack=self.lens_data.grid_stack,
                                                 border=self.lens_data.border, cosmology=self.cosmology)

        def padded_tracer_for_instance(self, instance):
            return ray_tracing.TracerMultiPlanes(galaxies=instance.galaxies,
                                                 image_plane_grid_stack=self.lens_data.padded_grid_stack,
                                                 cosmology=self.cosmology)

        @classmethod
        def log(cls, instance):
            logger.debug("\nRunning multi-plane for... \n\nGalaxies:\n{}\n\n".format(instance.galaxies))


class GalaxyFitPhase(AbstractPhase):

    galaxy = PhasePropertyCollection("galaxy")

    def __init__(self, phase_name, galaxy=None, use_intensities=False, use_surface_density=False, use_potential=False,
                 use_deflections=False, optimizer_class=non_linear.MultiNest, sub_grid_size=2,
                 mask_function=None, cosmology=cosmo.Planck15):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper
        passed to it.

        Parameters
        ----------
        galaxy_data_class: class<gd.GalaxyData>
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(GalaxyFitPhase, self).__init__(optimizer_class=optimizer_class, cosmology=cosmology,
                                             phase_name=phase_name)
        self.use_intensities = use_intensities
        self.use_surface_density = use_surface_density
        self.use_potential = use_potential
        self.use_deflections = use_deflections
        self.galaxy = galaxy
        self.sub_grid_size = sub_grid_size
        self.mask_function = mask_function

    def run(self, galaxy_data, previous_results=None, mask=None):
        """
        Run this phase.

        Parameters
        ----------
        mask: Mask
            The default masks passed in by the pipeline
        noise_map
        array
        previous_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper.
        """
        analysis = self.make_analysis(galaxy_data=galaxy_data, previous_results=previous_results, mask=mask)
        result = self.run_analysis(analysis)

        return self.make_result(result, analysis)

    def make_analysis(self, galaxy_data, previous_results=None, mask=None):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        mask: Mask
            The default masks passed in by the pipeline
        array
        noise_map
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens: Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """

        mask = setup_phase_mask(data=galaxy_data, mask=mask, mask_function=self.mask_function,
                                inner_circular_mask_radii=None)

        self.pass_priors(previous_results)

        if self.use_intensities or self.use_surface_density or self.use_potential:

            galaxy_data = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask, sub_grid_size=self.sub_grid_size,
                                           use_intensities=self.use_intensities,
                                           use_surface_density=self.use_surface_density,
                                           use_potential=self.use_potential,
                                           use_deflections_y=self.use_deflections,
                                           use_deflections_x=self.use_deflections)

            return self.__class__.AnalysisSingle(galaxy_data=galaxy_data, phase_name=self.phase_name,
                                                 cosmology=self.cosmology, previous_results=previous_results)

        elif self.use_deflections:

            galaxy_data_y = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask, sub_grid_size=self.sub_grid_size,
                                             use_intensities=self.use_intensities,
                                             use_surface_density=self.use_surface_density,
                                             use_potential=self.use_potential,
                                             use_deflections_y=self.use_deflections, use_deflections_x=False)

            galaxy_data_x = gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask, sub_grid_size=self.sub_grid_size,
                                             use_intensities=self.use_intensities,
                                             use_surface_density=self.use_surface_density,
                                             use_potential=self.use_potential,
                                             use_deflections_y=False, use_deflections_x=self.use_deflections)

            return self.__class__.AnalysisDeflections(galaxy_data_y=galaxy_data_y, galaxy_data_x=galaxy_data_x,
                                                      cosmology=self.cosmology, phase_name=self.phase_name,
                                                      previous_results=previous_results)

    class Analysis(Phase.Analysis):

        def __init__(self, cosmology, phase_name, previous_results):
            super(GalaxyFitPhase.Analysis, self).__init__(cosmology=cosmology, phase_name=phase_name,
                                                          previous_results=previous_results)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning galaxy fit for... \n\nGalaxy::\n{}\n\n".format(instance.galaxy))

    # noinspection PyAbstractClass
    class AnalysisSingle(Analysis):

        def __init__(self, galaxy_data, cosmology, phase_name, previous_results=None):
            super(GalaxyFitPhase.AnalysisSingle, self).__init__(cosmology=cosmology, phase_name=phase_name,
                                                                previous_results=previous_results)

            self.galaxy_data = galaxy_data

        def fit(self, instance):
            fit = self.fit_for_instance(instance=instance)
            return fit.figure_of_merit

        def visualize(self, instance, suffix, during_analysis):
            self.plot_count += 1
            fit = self.fit_for_instance(instance)

            galaxy_fitting_plotters.plot_single_subplot(fit=fit, output_path=self.output_image_path,
                                                        output_format='png', ignore_config=False)

            return fit

        def fit_for_instance(self, instance):
            """
            Determine the fit of a lens galaxy and source galaxy to the lens_data in this lens.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model lens_data itself
            """
            return galaxy_fit.GalaxyFit(galaxy_data=self.galaxy_data, model_galaxy=instance.galaxy[0])

    # noinspection PyAbstractClass
    class AnalysisDeflections(Analysis):

        def __init__(self, galaxy_data_y, galaxy_data_x, cosmology, phase_name, previous_results=None):
            super(GalaxyFitPhase.AnalysisDeflections, self).__init__(cosmology=cosmology, phase_name=phase_name,
                                                                     previous_results=previous_results)

            self.galaxy_data_y = galaxy_data_y
            self.galaxy_data_x = galaxy_data_x

        def fit(self, instance):
            fit_y, fit_x = self.fit_for_instance(instance=instance)
            return fit_y.figure_of_merit + fit_x.figure_of_merit

        def visualize(self, instance, suffix, during_analysis):
            fit_y, fit_x = self.fit_for_instance(instance)

            galaxy_fitting_plotters.plot_single_subplot(fit=fit_y, output_path=self.output_image_path,
                                                        output_format='png', ignore_config=False)

            galaxy_fitting_plotters.plot_single_subplot(fit=fit_x, output_path=self.output_image_path,
                                                        output_format='png', ignore_config=False)

            return fit_y, fit_x

        def fit_for_instance(self, instance):
            fit_y = galaxy_fit.GalaxyFit(galaxy_data=self.galaxy_data_y, model_galaxy=instance.galaxy)
            fit_x = galaxy_fit.GalaxyFit(galaxy_data=self.galaxy_data_x, model_galaxy=instance.galaxy)

            return fit_y, fit_x

    class Result(Phase.Result):

        def __init__(self, constant, figure_of_merit, variable, analysis, optimizer):
            """
            The result of a phase
            """

            super(GalaxyFitPhase.Result, self).__init__(constant=constant, figure_of_merit=figure_of_merit,
                                                        variable=variable, analysis=analysis, optimizer=optimizer)


class SensitivityPhase(PhaseImaging):

    lens_galaxies = PhasePropertyCollection("lens_galaxies")
    source_galaxies = PhasePropertyCollection("source_galaxies")
    sensitive_galaxies = PhasePropertyCollection("sensitive_galaxies")

    def __init__(self, phase_name, lens_galaxies=None, source_galaxies=None, sensitive_galaxies=None,
                 optimizer_class=non_linear.MultiNest, sub_grid_size=2, mask_function=None,
                 cosmology=cosmo.Planck15):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(SensitivityPhase, self).__init__(optimizer_class=optimizer_class, sub_grid_size=sub_grid_size,
                                               mask_function=mask_function, cosmology=cosmology,
                                               phase_name=phase_name)

        self.lens_galaxies = lens_galaxies or []
        self.source_galaxies = source_galaxies or []
        self.sensitive_galaxies = sensitive_galaxies or []

    # noinspection PyAbstractClass
    class Analysis(PhaseImaging.Analysis):

        def __init__(self, lens_data, cosmology, phase_name, previous_results=None):
            self.lens_data = lens_data
            super(PhaseImaging.Analysis, self).__init__(cosmology=cosmology, phase_name=phase_name,
                                                        previous_results=previous_results)

        def fit(self, instance):
            """
            Determine the fit of a lens galaxy and source galaxy to the lens_data in this lens.

            Parameters
            ----------
            instance
                A model instance with attributes

            Returns
            -------
            fit: Fit
                A fractional value indicating how well this model fit and the model lens_data itself
            """
            tracer_normal = self.tracer_normal_for_instance(instance)
            tracer_sensitive = self.tracer_sensitive_for_instance(instance)
            fit = self.fit_for_tracers(tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)
            return fit.figure_of_merit

        def visualize(self, instance, suffix, during_analysis):

            self.plot_count += 1

            tracer_normal = self.tracer_normal_for_instance(instance)
            tracer_sensitive = self.tracer_sensitive_for_instance(instance)
            fit = self.fit_for_tracers(tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)

            ccd_plotters.plot_ccd_subplot(ccd_data=self.lens_data.ccd_data, mask=self.lens_data.mask,
                                          positions=self.lens_data.positions,
                                          output_path=self.output_image_path, output_format='png',
                                          ignore_config=False)

            ccd_plotters.plot_ccd_individual(ccd_data=self.lens_data.ccd_data, mask=self.lens_data.mask,
                                             positions=self.lens_data.positions,
                                             output_path=self.output_image_path,
                                             output_format='png')

            ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer_normal, output_path=self.output_image_path,
                                                          output_format='png', output_filename='tracer_normal',
                                                          ignore_config=False)

            ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer_sensitive, output_path=self.output_image_path,
                                                          output_format='png', output_filename='tracer_sensitive',
                                                          ignore_config=False)

            sensitivity_fit_plotters.plot_fit_subplot(fit=fit, output_path=self.output_image_path, output_format='png')

            return fit

        def tracer_normal_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(lens_galaxies=instance.lens_galaxies,
                                                       source_galaxies=instance.source_galaxies,
                                                       image_plane_grid_stack=self.lens_data.grid_stack,
                                                       border=self.lens_data.border)

        def tracer_sensitive_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(
                lens_galaxies=instance.lens_galaxies + instance.sensitive_galaxies,
                source_galaxies=instance.source_galaxies,
                image_plane_grid_stack=self.lens_data.grid_stack,
                border=self.lens_data.border)

        def fit_for_tracers(self, tracer_normal, tracer_sensitive):
            return sensitivity_fit.fit_lens_data_with_sensitivity_tracers(lens_data=self.lens_data,
                                                                          tracer_normal=tracer_normal,
                                                                          tracer_sensitive=tracer_sensitive)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning lens/source lens for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n Sensitive "
                "Galaxy\n{}\n\n "
                "".format(instance.lens_galaxies, instance.source_galaxies, instance.sensitive_galaxies))


def make_path_if_does_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
