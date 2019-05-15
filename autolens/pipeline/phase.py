import copy

import functools
import numpy as np
from astropy import cosmology as cosmo

from autofit import conf
from autofit.optimize import non_linear
from autofit.tools import phase as autofit_phase
from autofit.tools.phase_property import PhaseProperty
from autolens import exc
from autolens.data.array import mask as msk
from autolens.data.plotters import ccd_plotters
from autolens.lens import lens_data as ld, lens_fit
from autolens.lens import ray_tracing
from autolens.lens import sensitivity_fit
from autolens.lens.plotters import sensitivity_fit_plotters, ray_tracing_plotters, lens_fit_plotters
from autolens.model.galaxy import galaxy as g, galaxy_fit, galaxy_data as gd
from autolens.model.galaxy.plotters import galaxy_fit_plotters
from autolens.pipeline import tagging as tag


# from autolens.lens.summary import tracer_summary

def default_mask_function(image):
    return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=3.0)


def setup_phase_mask(data, mask, mask_function, inner_mask_radii):
    if mask_function is not None:
        mask = mask_function(image=data.image)
    elif mask is None and mask_function is None:
        mask = default_mask_function(image=data.image)

    if inner_mask_radii is not None:
        inner_mask = msk.Mask.circular(shape=mask.shape, pixel_scale=mask.pixel_scale,
                                       radius_arcsec=inner_mask_radii, invert=True)
        mask = mask + inner_mask

    return mask


class AbstractPhase(autofit_phase.AbstractPhase):

    def __init__(self, phase_name, phase_tag=None, phase_folders=None, tag_phases=True,
                 optimizer_class=non_linear.MultiNest,
                 cosmology=cosmo.Planck15, auto_link_priors=False):
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

        super().__init__(phase_name=phase_name, phase_tag=phase_tag, phase_folders=phase_folders, tag_phases=tag_phases,
                         optimizer_class=optimizer_class, auto_link_priors=auto_link_priors)

        self.cosmology = cosmology

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
    def phase_property_collections(self):
        """
        Returns
        -------
        phase_property_collections: [PhaseProperty]
            A list of phase property collections associated with this phase. This is used in automated prior passing and
            should be overridden for any phase that contains its own PhasePropertys.
        """
        return []

    @property
    def path(self):
        return self.optimizer.path

    @property
    def doc(self):
        if self.__doc__ is not None:
            return self.__doc__.replace("  ", "").replace("\n", " ")

    def pass_priors(self, results):
        """
        Perform any prior or constant passing. This could involve setting model attributes equal to priors or constants
        from a previous phase.

        Parameters
        ----------
        results: autofit.tools.pipeline.ResultsCollection
            The result of the previous phase
        """
        pass

    # noinspection PyAbstractClass
    class Analysis(non_linear.Analysis):

        def __init__(self, cosmology, results=None):
            """
            An lens object

            Parameters
            ----------
            results: autofit.tools.pipeline.ResultsCollection
                The results of all previous phases
            """

            self.results = results
            self.cosmology = cosmology

            self.plot_count = 0

        @property
        def last_results(self):
            if self.results is not None:
                return self.results.last

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
                                     previous_variable=result.previous_variable, gaussian_tuples=result.gaussian_tuples,
                                     analysis=analysis, optimizer=self.optimizer)

    class Result(non_linear.Result):

        def __init__(self, constant, figure_of_merit, previous_variable, gaussian_tuples, analysis, optimizer):
            """
            The result of a phase
            """
            super(Phase.Result, self).__init__(constant=constant, figure_of_merit=figure_of_merit,
                                               previous_variable=previous_variable, gaussian_tuples=gaussian_tuples)

            self.analysis = analysis
            self.optimizer = optimizer

            # summary_file = open(optimizer.phase_output_path + 'model.summary', mode='w+')
            # tracer_summary.summarize_tracer(summary_file=summary_file, tracer=self.most_likely_tracer,
            #                                 radii=[10.0, 500.0])
            # summary_file.close()

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

        def unmasked_image_for_galaxy(self, galaxy):
            return self.most_likely_fit.unmasked_model_image_for_galaxy(galaxy)


class Phase(AbstractPhase):

    def run(self, image, results=None, mask=None):
        raise NotImplementedError()

    # noinspection PyAbstractClass
    class Analysis(AbstractPhase.Analysis):

        def __init__(self, cosmology, results=None):
            super(Phase.Analysis, self).__init__(cosmology=cosmology, results=results)

            self.should_plot_mask = \
                conf.instance.general.get('output', 'plot_mask_on_images', bool)
            self.extract_array_from_mask = \
                conf.instance.general.get('output', 'extract_images_from_mask', bool)
            self.zoom_around_mask = \
                conf.instance.general.get('output', 'zoom_around_mask_of_images', bool)
            self.should_plot_positions = \
                conf.instance.general.get('output', 'plot_positions_on_images', bool)
            self.plot_units = \
                conf.instance.general.get('output', 'plot_units', str).strip()

            self.plot_ray_tracing_all_at_end_png = \
                conf.instance.general.get('output', 'plot_ray_tracing_all_at_end_png', bool)
            self.plot_ray_tracing_all_at_end_fits = \
                conf.instance.general.get('output', 'plot_ray_tracing_all_at_end_fits', bool)

            self.plot_ray_tracing_as_subplot = \
                conf.instance.general.get('output', 'plot_ray_tracing_as_subplot', bool)
            self.plot_ray_tracing_image_plane_image = \
                conf.instance.general.get('output', 'plot_ray_tracing_image_plane_image', bool)
            self.plot_ray_tracing_source_plane = \
                conf.instance.general.get('output', 'plot_ray_tracing_source_plane_image', bool)
            self.plot_ray_tracing_convergence = \
                conf.instance.general.get('output', 'plot_ray_tracing_convergence', bool)
            self.plot_ray_tracing_potential = \
                conf.instance.general.get('output', 'plot_ray_tracing_potential', bool)
            self.plot_ray_tracing_deflections = \
                conf.instance.general.get('output', 'plot_ray_tracing_deflections', bool)


class PhasePositions(AbstractPhase):

    lens_galaxies = PhaseProperty("lens_galaxies")

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies]

    def __init__(self, phase_name, tag_phases=True, phase_folders=None, lens_galaxies=None,
                 optimizer_class=non_linear.MultiNest,
                 cosmology=cosmo.Planck15, auto_link_priors=False):
        super().__init__(phase_name=phase_name, phase_tag=None, phase_folders=phase_folders, tag_phases=tag_phases,
                         optimizer_class=optimizer_class, cosmology=cosmology, auto_link_priors=auto_link_priors)
        self.lens_galaxies = lens_galaxies

    def run(self, positions, pixel_scale, results=None):
        """
        Run this phase.

        Parameters
        ----------
        pixel_scale
        positions
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper.
        """
        analysis = self.make_analysis(positions=positions, pixel_scale=pixel_scale, results=results)
        result = self.run_analysis(analysis)
        return self.make_result(result, analysis)

    def make_analysis(self, positions, pixel_scale, results=None):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        pixel_scale
        positions
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens: Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """
        self.pass_priors(results)
        analysis = self.__class__.Analysis(positions=positions, pixel_scale=pixel_scale, cosmology=self.cosmology,
                                           results=results)
        return analysis

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, positions, pixel_scale, cosmology, results=None):
            super().__init__(cosmology=cosmology, results=results)

            self.positions = list(map(lambda position_set: np.asarray(position_set), positions))
            self.pixel_scale = pixel_scale

        def visualize(self, instance, image_path, during_analysis):
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
        def describe(cls, instance):
            return "\nRunning lens lens for... \n\nLens Galaxy::\n{}\n\n".format(instance.lens_galaxies)


class PhaseImaging(Phase):

    def __init__(self, phase_name, tag_phases=True, phase_folders=None, optimizer_class=non_linear.MultiNest,
                 sub_grid_size=2, bin_up_factor=None, image_psf_shape=None,
                 inversion_psf_shape=None, positions_threshold=None, mask_function=None, inner_mask_radii=None,
                 interp_pixel_scale=None, cosmology=cosmo.Planck15, auto_link_priors=False, uses_inversion=True):

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

        if tag_phases:

            phase_tag = tag.phase_tag_from_phase_settings(sub_grid_size=sub_grid_size,
                                                          bin_up_factor=bin_up_factor,
                                                          image_psf_shape=image_psf_shape,
                                                          inversion_psf_shape=inversion_psf_shape,
                                                          positions_threshold=positions_threshold,
                                                          inner_mask_radii=inner_mask_radii,
                                                          interp_pixel_scale=interp_pixel_scale)

        else:

            phase_tag = None

        super(PhaseImaging, self).__init__(phase_name=phase_name, phase_tag=phase_tag, phase_folders=phase_folders,
                                           tag_phases=tag_phases,
                                           optimizer_class=optimizer_class, cosmology=cosmology,
                                           auto_link_priors=auto_link_priors)
        self.sub_grid_size = sub_grid_size
        self.bin_up_factor = bin_up_factor
        self.image_psf_shape = image_psf_shape
        self.inversion_psf_shape = inversion_psf_shape
        self.positions_threshold = positions_threshold
        self.mask_function = mask_function
        self.inner_mask_radii = inner_mask_radii
        self.interp_pixel_scale = interp_pixel_scale
        self.uses_inversion = uses_inversion

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def modify_image(self, image, results):
        """
        Customize an lens_data. e.g. removing lens light.

        Parameters
        ----------
        image: scaled_array.ScaledSquarePixelArray
            An lens_data that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result of the previous lens

        Returns
        -------
        lens_data: scaled_array.ScaledSquarePixelArray
            The modified image (not changed by default)
        """
        return image

    def run(self, data, results=None, mask=None, positions=None):
        """
        Run this phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        data: scaled_array.ScaledSquarePixelArray
            An lens_data that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper.
        """
        analysis = self.make_analysis(data=data, results=results, mask=mask, positions=positions)

        result = self.run_analysis(analysis)

        return self.make_result(result, analysis)

    def make_analysis(self, data, results=None, mask=None, positions=None):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        data: im.CCD
            An lens_data that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """

        mask = setup_phase_mask(data=data, mask=mask, mask_function=self.mask_function,
                                inner_mask_radii=self.inner_mask_radii)

        if self.positions_threshold is not None and positions is not None:
            positions = list(map(lambda position_set: np.asarray(position_set), positions))
        elif self.positions_threshold is None:
            positions = None
        elif self.positions_threshold is not None and positions is None:
            raise exc.PhaseException('You have specified for a phase to use positions, but not input positions to the '
                                     'pipeline when you ran it.')

        lens_data = ld.LensData(ccd_data=data, mask=mask, sub_grid_size=self.sub_grid_size,
                                image_psf_shape=self.image_psf_shape, positions=positions,
                                interp_pixel_scale=self.interp_pixel_scale, uses_inversion=self.uses_inversion)

        modified_image = self.modify_image(image=lens_data.image, results=results)
        lens_data = lens_data.new_lens_data_with_modified_image(modified_image=modified_image)

        if self.bin_up_factor is not None:
            lens_data = lens_data.new_lens_data_with_binned_up_ccd_data_and_mask(bin_up_factor=self.bin_up_factor)

        self.pass_priors(results)

        self.output_phase_info()

        analysis = self.__class__.Analysis(lens_data=lens_data, cosmology=self.cosmology,
                                           positions_threshold=self.positions_threshold, results=results)
        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(self.optimizer.phase_output_path, 'phase.info')

        with open(file_phase_info, 'w') as phase_info:
            phase_info.write('Optimizer = {} \n'.format(type(self.optimizer).__name__))
            phase_info.write('Sub-grid size = {} \n'.format(self.sub_grid_size))
            phase_info.write('Image PSF shape = {} \n'.format(self.image_psf_shape))
            phase_info.write('Pixelization PSF shape = {} \n'.format(self.inversion_psf_shape))
            phase_info.write('Positions Threshold = {} \n'.format(self.positions_threshold))
            phase_info.write('Cosmology = {} \n'.format(self.cosmology))
            phase_info.write('Auto Link Priors = {} \n'.format(self.auto_link_priors))

            phase_info.close()

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, lens_data, cosmology, positions_threshold, results=None):

            super(PhaseImaging.Analysis, self).__init__(cosmology=cosmology, results=results)

            self.lens_data = lens_data

            self.positions_threshold = positions_threshold

            self.should_plot_image_plane_pix = \
                conf.instance.general.get('output', 'plot_image_plane_adaptive_pixelization_grid', bool)

            self.plot_data_as_subplot = \
                conf.instance.general.get('output', 'plot_data_as_subplot', bool)
            self.plot_data_image = \
                conf.instance.general.get('output', 'plot_data_image', bool)
            self.plot_data_noise_map = \
                conf.instance.general.get('output', 'plot_data_noise_map', bool)
            self.plot_data_psf = \
                conf.instance.general.get('output', 'plot_data_psf', bool)
            self.plot_data_signal_to_noise_map = \
                conf.instance.general.get('output', 'plot_data_signal_to_noise_map', bool)
            self.plot_data_absolute_signal_to_noise_map = \
                conf.instance.general.get('output', 'plot_data_absolute_signal_to_noise_map', bool)
            self.plot_data_potential_chi_squared_map = \
                conf.instance.general.get('output', 'plot_data_potential_chi_squared_map', bool)

            self.plot_lens_fit_all_at_end_png = \
                conf.instance.general.get('output', 'plot_lens_fit_all_at_end_png', bool)
            self.plot_lens_fit_all_at_end_fits = \
                conf.instance.general.get('output', 'plot_lens_fit_all_at_end_fits', bool)

            self.plot_lens_fit_as_subplot = \
                conf.instance.general.get('output', 'plot_lens_fit_as_subplot', bool)
            self.plot_lens_fit_image = \
                conf.instance.general.get('output', 'plot_lens_fit_image', bool)
            self.plot_lens_fit_noise_map = \
                conf.instance.general.get('output', 'plot_lens_fit_noise_map', bool)
            self.plot_lens_fit_signal_to_noise_map = \
                conf.instance.general.get('output', 'plot_lens_fit_signal_to_noise_map', bool)
            self.plot_lens_fit_lens_subtracted_image = \
                conf.instance.general.get('output', 'plot_lens_fit_lens_subtracted_image', bool)
            self.plot_lens_fit_model_image = \
                conf.instance.general.get('output', 'plot_lens_fit_model_image', bool)
            self.plot_lens_fit_lens_model_image = \
                conf.instance.general.get('output', 'plot_lens_fit_lens_model_image', bool)
            self.plot_lens_fit_source_model_image = \
                conf.instance.general.get('output', 'plot_lens_fit_source_model_image', bool)
            self.plot_lens_fit_source_plane_image = \
                conf.instance.general.get('output', 'plot_lens_fit_source_plane_image', bool)
            self.plot_lens_fit_residual_map = \
                conf.instance.general.get('output', 'plot_lens_fit_residual_map', bool)
            self.plot_lens_fit_chi_squared_map = \
                conf.instance.general.get('output', 'plot_lens_fit_chi_squared_map', bool)
            self.plot_lens_fit_contribution_map = \
                conf.instance.general.get('output', 'plot_lens_fit_contribution_map', bool)

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

        def visualize(self, instance, image_path, during_analysis):

            mask = self.lens_data.mask if self.should_plot_mask else None
            positions = self.lens_data.positions if self.should_plot_positions else None

            ccd_plotters.plot_ccd_for_phase(
                ccd_data=self.lens_data.ccd_data, mask=mask, positions=positions,
                extract_array_from_mask=self.extract_array_from_mask, zoom_around_mask=self.zoom_around_mask,
                units=self.plot_units,
                should_plot_as_subplot=self.plot_data_as_subplot,
                should_plot_image=self.plot_data_image,
                should_plot_noise_map=self.plot_data_noise_map,
                should_plot_psf=self.plot_data_psf,
                should_plot_signal_to_noise_map=self.plot_data_signal_to_noise_map,
                should_plot_absolute_signal_to_noise_map=self.plot_data_absolute_signal_to_noise_map,
                should_plot_potential_chi_squared_map=self.plot_data_potential_chi_squared_map,
                visualize_path=image_path)

            tracer = self.tracer_for_instance(instance)

            ray_tracing_plotters.plot_ray_tracing_for_phase(
                tracer=tracer, during_analysis=during_analysis, mask=mask,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask, positions=positions, units=self.plot_units,
                should_plot_as_subplot=self.plot_ray_tracing_as_subplot,
                should_plot_all_at_end_png=self.plot_ray_tracing_all_at_end_png,
                should_plot_all_at_end_fits=self.plot_ray_tracing_all_at_end_fits,
                should_plot_image_plane_image=self.plot_ray_tracing_image_plane_image,
                should_plot_source_plane=self.plot_ray_tracing_source_plane,
                should_plot_convergence=self.plot_ray_tracing_convergence,
                should_plot_potential=self.plot_ray_tracing_potential,
                should_plot_deflections=self.plot_ray_tracing_deflections,
                visualize_path=image_path)

            padded_tracer = self.padded_tracer_for_instance(instance)
            fit = self.fit_for_tracers(tracer=tracer, padded_tracer=padded_tracer)

            lens_fit_plotters.plot_lens_fit_for_phase(
                fit=fit, during_analysis=during_analysis, should_plot_mask=self.should_plot_mask,
                extract_array_from_mask=self.extract_array_from_mask, zoom_around_mask=self.zoom_around_mask,
                positions=positions, should_plot_image_plane_pix=self.should_plot_image_plane_pix,
                should_plot_as_subplot=self.plot_lens_fit_as_subplot,
                should_plot_all_at_end_png=self.plot_lens_fit_all_at_end_png,
                should_plot_all_at_end_fits=self.plot_lens_fit_all_at_end_fits,
                should_plot_image=self.plot_lens_fit_image,
                should_plot_noise_map=self.plot_lens_fit_noise_map,
                should_plot_signal_to_noise_map=self.plot_lens_fit_signal_to_noise_map,
                should_plot_lens_subtracted_image=self.plot_lens_fit_lens_subtracted_image,
                should_plot_model_image=self.plot_lens_fit_model_image,
                should_plot_lens_model_image=self.plot_lens_fit_lens_model_image,
                should_plot_source_model_image=self.plot_lens_fit_source_model_image,
                should_plot_source_plane_image=self.plot_lens_fit_source_plane_image,
                should_plot_residual_map=self.plot_lens_fit_residual_map,
                should_plot_chi_squared_map=self.plot_lens_fit_chi_squared_map,
                units=self.plot_units,
                visualize_path=image_path)

        def fit_for_tracers(self, tracer, padded_tracer):
            return lens_fit.LensDataFit.for_data_and_tracer(lens_data=self.lens_data, tracer=tracer,
                                                            padded_tracer=padded_tracer)

        def check_positions_trace_within_threshold(self, instance):

            if self.lens_data.positions is not None:

                tracer = ray_tracing.TracerImageSourcePlanesPositions(lens_galaxies=instance.lens_galaxies,
                                                                      image_plane_positions=self.lens_data.positions)
                fit = lens_fit.LensPositionFit(positions=tracer.source_plane.positions,
                                               noise_map=self.lens_data.pixel_scale)

                if not fit.maximum_separation_within_threshold(self.positions_threshold):
                    raise exc.RayTracingException

        def map_to_1d(self, data):
            """Convenience method"""
            return self.lens_data.mask.map_2d_array_to_masked_1d_array(data)


def set_defaults(key):
    """
    Load a default value for redshift from config and set it as the redshift for source or lens galaxies that have
    falsey redshifts

    Parameters
    ----------
    key: str

    Returns
    -------
    decorator
        A decorator that wraps the setter function to set defaults
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(phase, new_value):
            new_value = new_value or []
            for item in new_value:
                # noinspection PyTypeChecker
                galaxy = new_value[item] if isinstance(item, str) else item
                galaxy.redshift = galaxy.redshift or conf.instance.general.get("redshift", key, float)
            return func(phase, new_value)

        return wrapper

    return decorator


class LensPlanePhase(PhaseImaging):
    """
    Fit only the lens galaxy light.
    """

    _lens_galaxies = PhaseProperty("lens_galaxies")

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies]

    def __init__(self, phase_name, tag_phases=True, phase_folders=None, lens_galaxies=None,
                 optimizer_class=non_linear.MultiNest,
                 sub_grid_size=2, bin_up_factor=None,
                 image_psf_shape=None, mask_function=None, inner_mask_radii=None, cosmology=cosmo.Planck15,
                 auto_link_priors=False):

        uses_inversion = False

        if isinstance(lens_galaxies, dict):
            for key, galaxy_model in lens_galaxies.items():
                if galaxy_model.pixelization is not None:
                    uses_inversion = True

        super(LensPlanePhase, self).__init__(phase_name=phase_name,
                                             tag_phases=tag_phases,
                                             phase_folders=phase_folders,
                                             optimizer_class=optimizer_class,
                                             sub_grid_size=sub_grid_size,
                                             bin_up_factor=bin_up_factor,
                                             image_psf_shape=image_psf_shape,
                                             mask_function=mask_function,
                                             inner_mask_radii=inner_mask_radii,
                                             cosmology=cosmology,
                                             auto_link_priors=auto_link_priors,
                                             uses_inversion=uses_inversion)
        self.lens_galaxies = lens_galaxies

    @property
    def lens_galaxies(self):
        return self._lens_galaxies

    @lens_galaxies.setter
    @set_defaults("lens_default")
    def lens_galaxies(self, new_value):
        self._lens_galaxies = new_value

    class Analysis(PhaseImaging.Analysis):
        def figure_of_merit_for_fit(self, tracer):
            raise NotImplementedError()

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerImagePlane(lens_galaxies=instance.lens_galaxies,
                                                image_plane_grid_stack=self.lens_data.grid_stack,
                                                cosmology=self.cosmology)

        def padded_tracer_for_instance(self, instance):
            return ray_tracing.TracerImagePlane(lens_galaxies=instance.lens_galaxies,
                                                image_plane_grid_stack=self.lens_data.padded_grid_stack,
                                                cosmology=self.cosmology)

        @classmethod
        def describe(cls, instance):
            return "\nRunning lens lens for... \n\nLens Galaxy::\n{}\n\n".format(instance.lens_galaxies)

    class Result(PhaseImaging.Result):
        @property
        def unmasked_lens_plane_model_image(self):
            return self.most_likely_fit.unmasked_model_image_of_planes[0]


class LensSourcePlanePhase(PhaseImaging):
    """
    Fit a simple source and lens system.
    """

    _lens_galaxies = PhaseProperty("lens_galaxies")
    _source_galaxies = PhaseProperty("source_galaxies")

    @property
    def lens_galaxies(self):
        return self._lens_galaxies

    @lens_galaxies.setter
    @set_defaults("lens_default")
    def lens_galaxies(self, new_value):
        self._lens_galaxies = new_value

    @property
    def source_galaxies(self):
        return self._source_galaxies

    @source_galaxies.setter
    @set_defaults("source_default")
    def source_galaxies(self, new_value):
        self._source_galaxies = new_value

    @property
    def phase_property_collections(self):
        return [self.lens_galaxies, self.source_galaxies]

    def __init__(self, phase_name, tag_phases=True, phase_folders=None,
                 lens_galaxies=None, source_galaxies=None, optimizer_class=non_linear.MultiNest,
                 sub_grid_size=2, bin_up_factor=None, image_psf_shape=None, positions_threshold=None,
                 mask_function=None,
                 interp_pixel_scale=None, inner_mask_radii=None, cosmology=cosmo.Planck15,
                 auto_link_priors=False):
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

        uses_inversion = False

        if isinstance(lens_galaxies, dict):
            for key, galaxy_model in lens_galaxies.items():
                if galaxy_model.pixelization is not None:
                    uses_inversion = True

        if isinstance(source_galaxies, dict):
            for key, galaxy_model in source_galaxies.items():
                if galaxy_model.pixelization is not None:
                    uses_inversion = True

        super(LensSourcePlanePhase, self).__init__(phase_name=phase_name,
                                                   tag_phases=tag_phases,
                                                   phase_folders=phase_folders,
                                                   optimizer_class=optimizer_class,
                                                   sub_grid_size=sub_grid_size,
                                                   bin_up_factor=bin_up_factor,
                                                   image_psf_shape=image_psf_shape,
                                                   positions_threshold=positions_threshold,
                                                   mask_function=mask_function,
                                                   interp_pixel_scale=interp_pixel_scale,
                                                   inner_mask_radii=inner_mask_radii,
                                                   cosmology=cosmology,
                                                   auto_link_priors=auto_link_priors,
                                                   uses_inversion=uses_inversion)
        self.lens_galaxies = lens_galaxies or []
        self.source_galaxies = source_galaxies or []

    class Analysis(PhaseImaging.Analysis):
        def figure_of_merit_for_fit(self, tracer):
            raise NotImplementedError()

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
        def describe(cls, instance):
            return "\nRunning lens/source lens for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n".format(
                instance.lens_galaxies, instance.source_galaxies)

    class Result(PhaseImaging.Result):

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

    galaxies = PhaseProperty("galaxies")

    @property
    def phase_property_collections(self):
        return [self.galaxies]

    def __init__(self, phase_name, tag_phases=True, phase_folders=None, galaxies=None,
                 optimizer_class=non_linear.MultiNest,
                 sub_grid_size=2, bin_up_factor=None, image_psf_shape=None, positions_threshold=None,
                 mask_function=None,
                 inner_mask_radii=None, cosmology=cosmo.Planck15, auto_link_priors=False):
        """
        A phase with a simple source/lens model

        Parameters
        ----------
        galaxies : [g.Galaxy] | [gm.GalaxyModel]
            A galaxy that acts as a gravitational lens or is being lensed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """

        super(MultiPlanePhase, self).__init__(phase_name=phase_name,
                                              tag_phases=tag_phases,
                                              phase_folders=phase_folders,
                                              optimizer_class=optimizer_class,
                                              sub_grid_size=sub_grid_size,
                                              bin_up_factor=bin_up_factor,
                                              image_psf_shape=image_psf_shape,
                                              positions_threshold=positions_threshold,
                                              mask_function=mask_function,
                                              inner_mask_radii=inner_mask_radii,
                                              cosmology=cosmology,
                                              auto_link_priors=auto_link_priors,
                                              uses_inversion=True)
        self.galaxies = galaxies

    class Analysis(PhaseImaging.Analysis):

        def figure_of_merit_for_fit(self, tracer):
            raise NotImplementedError()

        def __init__(self, lens_data, cosmology, positions_threshold, results=None):
            self.lens_data = lens_data
            super(MultiPlanePhase.Analysis, self).__init__(lens_data=lens_data, cosmology=cosmology,
                                                           positions_threshold=positions_threshold, results=results)

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerMultiPlanes(galaxies=instance.galaxies,
                                                 image_plane_grid_stack=self.lens_data.grid_stack,
                                                 border=self.lens_data.border, cosmology=self.cosmology)

        def padded_tracer_for_instance(self, instance):
            return ray_tracing.TracerMultiPlanes(galaxies=instance.galaxies,
                                                 image_plane_grid_stack=self.lens_data.padded_grid_stack,
                                                 cosmology=self.cosmology)

        @classmethod
        def describe(cls, instance):
            return "\nRunning multi-plane for... \n\nGalaxies:\n{}\n\n".format(instance.galaxies)


class GalaxyFitPhase(AbstractPhase):
    galaxies = PhaseProperty("galaxies")

    def __init__(self, phase_name, phase_tagging=True, phase_folders=None, galaxies=None, use_intensities=False,
                 use_convergence=False,
                 use_potential=False,
                 use_deflections=False, optimizer_class=non_linear.MultiNest, sub_grid_size=2,
                 mask_function=None, cosmology=cosmo.Planck15):
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

        super(GalaxyFitPhase, self).__init__(phase_name=phase_name, phase_tagging=phase_tagging,
                                             phase_folders=phase_folders,
                                             optimizer_class=optimizer_class, cosmology=cosmology)
        self.use_intensities = use_intensities
        self.use_convergence = use_convergence
        self.use_potential = use_potential
        self.use_deflections = use_deflections
        self.galaxies = galaxies
        self.sub_grid_size = sub_grid_size
        self.mask_function = mask_function

    def run(self, galaxy_data, results=None, mask=None):
        """
        Run this phase.

        Parameters
        ----------
        galaxy_data
        mask: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper.
        """
        analysis = self.make_analysis(galaxy_data=galaxy_data, results=results, mask=mask)
        result = self.run_analysis(analysis)

        return self.make_result(result, analysis)

    def make_analysis(self, galaxy_data, results=None, mask=None):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        galaxy_data
        mask: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens: Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """

        mask = setup_phase_mask(data=galaxy_data[0], mask=mask, mask_function=self.mask_function,
                                inner_mask_radii=None)

        self.pass_priors(results)

        if self.use_intensities or self.use_convergence or self.use_potential:

            galaxy_data = gd.GalaxyFitData(galaxy_data=galaxy_data[0], mask=mask, sub_grid_size=self.sub_grid_size,
                                           use_intensities=self.use_intensities,
                                           use_convergence=self.use_convergence,
                                           use_potential=self.use_potential,
                                           use_deflections_y=self.use_deflections,
                                           use_deflections_x=self.use_deflections)

            return self.__class__.AnalysisSingle(galaxy_data=galaxy_data,
                                                 cosmology=self.cosmology,
                                                 results=results)

        elif self.use_deflections:

            galaxy_data_y = gd.GalaxyFitData(galaxy_data=galaxy_data[0], mask=mask, sub_grid_size=self.sub_grid_size,
                                             use_intensities=self.use_intensities,
                                             use_convergence=self.use_convergence,
                                             use_potential=self.use_potential,
                                             use_deflections_y=self.use_deflections, use_deflections_x=False)

            galaxy_data_x = gd.GalaxyFitData(galaxy_data=galaxy_data[1], mask=mask, sub_grid_size=self.sub_grid_size,
                                             use_intensities=self.use_intensities,
                                             use_convergence=self.use_convergence,
                                             use_potential=self.use_potential,
                                             use_deflections_y=False, use_deflections_x=self.use_deflections)

            return self.__class__.AnalysisDeflections(galaxy_data_y=galaxy_data_y, galaxy_data_x=galaxy_data_x,
                                                      cosmology=self.cosmology,
                                                      results=results)

    class Analysis(Phase.Analysis):

        def __init__(self, cosmology, results):
            super(GalaxyFitPhase.Analysis, self).__init__(cosmology=cosmology,
                                                          results=results)

            self.plot_galaxy_fit_all_at_end_png = \
                conf.instance.general.get('output', 'plot_galaxy_fit_all_at_end_png', bool)
            self.plot_galaxy_fit_all_at_end_fits = \
                conf.instance.general.get('output', 'plot_galaxy_fit_all_at_end_fits', bool)
            self.plot_galaxy_fit_as_subplot = \
                conf.instance.general.get('output', 'plot_galaxy_fit_as_subplot', bool)
            self.plot_galaxy_fit_image = \
                conf.instance.general.get('output', 'plot_galaxy_fit_image', bool)
            self.plot_galaxy_fit_noise_map = \
                conf.instance.general.get('output', 'plot_galaxy_fit_noise_map', bool)
            self.plot_galaxy_fit_model_image = \
                conf.instance.general.get('output', 'plot_galaxy_fit_model_image', bool)
            self.plot_galaxy_fit_residual_map = \
                conf.instance.general.get('output', 'plot_galaxy_fit_residual_map', bool)
            self.plot_galaxy_fit_chi_squared_map = \
                conf.instance.general.get('output', 'plot_galaxy_fit_chi_squared_map', bool)

        @classmethod
        def describe(cls, instance):
            return "\nRunning galaxy fit for... \n\nGalaxies::\n{}\n\n".format(instance.galaxies)

    # noinspection PyAbstractClass
    class AnalysisSingle(Analysis):

        def __init__(self, galaxy_data, cosmology, results=None):
            super(GalaxyFitPhase.AnalysisSingle, self).__init__(cosmology=cosmology,
                                                                results=results)

            self.galaxy_data = galaxy_data

        def fit(self, instance):
            fit = self.fit_for_instance(instance=instance)
            return fit.figure_of_merit

        def visualize(self, instance, image_path, during_analysis):

            self.plot_count += 1
            fit = self.fit_for_instance(instance=instance)

            if self.plot_galaxy_fit_as_subplot:
                galaxy_fit_plotters.plot_fit_subplot(
                    fit=fit, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                    units=self.plot_units,
                    output_path=image_path, output_format='png')

            if during_analysis:

                galaxy_fit_plotters.plot_fit_individuals(
                    fit=fit, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                    should_plot_image=self.plot_galaxy_fit_image,
                    should_plot_noise_map=self.plot_galaxy_fit_noise_map,
                    should_plot_model_image=self.plot_galaxy_fit_model_image,
                    should_plot_residual_map=self.plot_galaxy_fit_residual_map,
                    should_plot_chi_squared_map=self.plot_galaxy_fit_chi_squared_map,
                    units=self.plot_units,
                    output_path=image_path, output_format='png')

            elif not during_analysis:

                if self.plot_ray_tracing_all_at_end_png:
                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path=image_path, output_format='png')

                if self.plot_ray_tracing_all_at_end_fits:
                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path="{}/fits/".format(image_path), output_format='fits')

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
            return galaxy_fit.GalaxyFit(galaxy_data=self.galaxy_data, model_galaxies=instance.galaxies)

    # noinspection PyAbstractClass
    class AnalysisDeflections(Analysis):

        def __init__(self, galaxy_data_y, galaxy_data_x, cosmology, results=None):
            super(GalaxyFitPhase.AnalysisDeflections, self).__init__(cosmology=cosmology,
                                                                     results=results)

            self.galaxy_data_y = galaxy_data_y
            self.galaxy_data_x = galaxy_data_x

        def fit(self, instance):
            fit_y, fit_x = self.fit_for_instance(instance=instance)
            return fit_y.figure_of_merit + fit_x.figure_of_merit

        def visualize(self, instance, image_path, during_analysis):

            output_image_y_path = "{}/fit_y_".format(image_path)
            output_fits_y_path = "{}/fits/fit_y".format(image_path)
            output_image_x_path = "{}/fit_x_".format(image_path)
            output_fits_x_path = "{}/fits/fit_x".format(image_path)

            self.plot_count += 1
            fit_y, fit_x = self.fit_for_instance(instance=instance)

            if self.plot_galaxy_fit_as_subplot:
                galaxy_fit_plotters.plot_fit_subplot(
                    fit=fit_y, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                    units=self.plot_units,
                    output_path=output_image_y_path, output_format='png')

                galaxy_fit_plotters.plot_fit_subplot(
                    fit=fit_x, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                    units=self.plot_units,
                    output_path=output_image_x_path, output_format='png')

            if during_analysis:

                galaxy_fit_plotters.plot_fit_individuals(
                    fit=fit_y, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                    should_plot_image=self.plot_galaxy_fit_image,
                    should_plot_noise_map=self.plot_galaxy_fit_noise_map,
                    should_plot_model_image=self.plot_galaxy_fit_model_image,
                    should_plot_residual_map=self.plot_galaxy_fit_residual_map,
                    should_plot_chi_squared_map=self.plot_galaxy_fit_chi_squared_map,
                    units=self.plot_units,
                    output_path=output_image_y_path, output_format='png')

                galaxy_fit_plotters.plot_fit_individuals(
                    fit=fit_x, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                    should_plot_image=self.plot_galaxy_fit_image,
                    should_plot_noise_map=self.plot_galaxy_fit_noise_map,
                    should_plot_model_image=self.plot_galaxy_fit_model_image,
                    should_plot_residual_map=self.plot_galaxy_fit_residual_map,
                    should_plot_chi_squared_map=self.plot_galaxy_fit_chi_squared_map,
                    units=self.plot_units,
                    output_path=output_image_x_path, output_format='png')

            elif not during_analysis:

                if self.plot_ray_tracing_all_at_end_png:
                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit_y, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path=output_image_y_path, output_format='png')

                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit_x, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path=output_image_x_path, output_format='png')

                if self.plot_ray_tracing_all_at_end_fits:
                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit_y, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path=output_fits_y_path, output_format='fits')

                    galaxy_fit_plotters.plot_fit_individuals(
                        fit=fit_x, should_plot_mask=self.should_plot_mask, zoom_around_mask=self.zoom_around_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_model_image=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        units=self.plot_units,
                        output_path=output_fits_x_path, output_format='fits')

            return fit_y, fit_x

        def fit_for_instance(self, instance):

            fit_y = galaxy_fit.GalaxyFit(galaxy_data=self.galaxy_data_y, model_galaxies=instance.galaxies)
            fit_x = galaxy_fit.GalaxyFit(galaxy_data=self.galaxy_data_x, model_galaxies=instance.galaxies)

            return fit_y, fit_x


class SensitivityPhase(PhaseImaging):
    lens_galaxies = PhaseProperty("lens_galaxies")
    source_galaxies = PhaseProperty("source_galaxies")
    sensitive_galaxies = PhaseProperty("sensitive_galaxies")

    def __init__(self, phase_name, tag_phases=None, phase_folders=None, lens_galaxies=None, source_galaxies=None,
                 sensitive_galaxies=None,
                 optimizer_class=non_linear.MultiNest, sub_grid_size=2, bin_up_factor=None, mask_function=None,
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

        super(SensitivityPhase, self).__init__(phase_name=phase_name, tag_phases=tag_phases,
                                               phase_folders=phase_folders,
                                               optimizer_class=optimizer_class, sub_grid_size=sub_grid_size,
                                               bin_up_factor=bin_up_factor, mask_function=mask_function,
                                               cosmology=cosmology)

        self.lens_galaxies = lens_galaxies or []
        self.source_galaxies = source_galaxies or []
        self.sensitive_galaxies = sensitive_galaxies or []

    # noinspection PyAbstractClass
    class Analysis(PhaseImaging.Analysis):

        def __init__(self, lens_data, cosmology, phase_name, results=None):
            self.lens_data = lens_data
            super(PhaseImaging.Analysis, self).__init__(cosmology=cosmology,
                                                        results=results)

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

        def visualize(self, instance, image_path, during_analysis):
            self.plot_count += 1

            tracer_normal = self.tracer_normal_for_instance(instance)
            tracer_sensitive = self.tracer_sensitive_for_instance(instance)
            fit = self.fit_for_tracers(tracer_normal=tracer_normal, tracer_sensitive=tracer_sensitive)

            ccd_plotters.plot_ccd_subplot(ccd_data=self.lens_data.ccd_data, mask=self.lens_data.mask,
                                          positions=self.lens_data.positions,
                                          output_path=image_path, output_format='png')

            ccd_plotters.plot_ccd_individual(ccd_data=self.lens_data.ccd_data, mask=self.lens_data.mask,
                                             positions=self.lens_data.positions,
                                             output_path=image_path,
                                             output_format='png')

            ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer_normal, output_path=image_path,
                                                          output_format='png', output_filename='tracer_normal')

            ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer_sensitive, output_path=image_path,
                                                          output_format='png', output_filename='tracer_sensitive')

            sensitivity_fit_plotters.plot_fit_subplot(fit=fit, output_path=image_path, output_format='png')

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
        def describe(cls, instance):
            return "\nRunning lens/source lens for... \n\nLens Galaxy:\n{}\n\nSource Galaxy:\n{}\n\n Sensitive " \
                   "Galaxy\n{}\n\n ".format(instance.lens_galaxies, instance.source_galaxies,
                                            instance.sensitive_galaxies)


class HyperGalaxyPhase(Phase):

    class Analysis(non_linear.Analysis):

        def __init__(self, lens_data, model_image, galaxy_image):
            """
            An analysis to fit the noise for a single galaxy image.

            Parameters
            ----------
            lens_data: LensData
                Lens data, including an image and noise
            model_image: ndarray
                An image produce of the overall system by a model
            galaxy_image: ndarray
                The contribution of one galaxy to the model image
            """
            self.lens_data = lens_data
            self.model_image = model_image
            self.galaxy_image = galaxy_image

        def visualize(self, instance, image_path, during_analysis):
            # TODO: I'm guessing you have an idea of what you want here?
            pass

        def fit(self, instance):
            """
            Fit the model image to the real image by scaling the hyper noise.

            Parameters
            ----------
            instance: ModelInstance
                A model instance with a hyper galaxy property

            Returns
            -------
            fit: float
            """
            fit = self.fit_for_hyper_galaxy(hyper_galaxy=instance.hyper_galaxy)
            return fit.figure_of_merit

        def fit_for_hyper_galaxy(self, hyper_galaxy):
            hyper_noise = hyper_galaxy.hyper_noise_from_model_image_galaxy_image_and_noise_map(self.model_image,
                                                                                               self.galaxy_image,
                                                                                               self.lens_data.noise_map)
            hyper_noise_map = self.lens_data.noise_map + hyper_noise
            return lens_fit.LensDataFit(image=self.lens_data.image, noise_map=hyper_noise_map,
                                        mask=np.full(self.lens_data.image.shape, False), model_image=self.model_image)

        @classmethod
        def describe(cls, instance):
            return "Running hyper galaxy fit for HyperGalaxy:\n{}".format(instance.hyper_galaxy)

    def run(self, data, results=None, mask=None, positions=None):
        """
        Run a fit for each galaxy from the previous phase.

        Parameters
        ----------
        data: LensData
        results: ResultsCollection
            Results from all previous phases
        mask: Mask
            The mask
        positions

        Returns
        -------
        results: HyperGalaxyResults
            A collection of results, with one item per a galaxy
        """
        model_image = results.last.unmasked_model_image
        galaxy_tuples = results.last.constant.name_instance_tuples_for_class(g.Galaxy)

        results_copy = copy.copy(results.last)

        for name, galaxy in galaxy_tuples:
            optimizer = self.optimizer.copy_with_name_extension(name)
            optimizer.variable.hyper_galaxy = g.HyperGalaxy
            galaxy_image = results.last.unmasked_image_for_galaxy(galaxy)
            optimizer.fit(self.__class__.Analysis(data, model_image, galaxy_image))

            getattr(results_copy.variable, name).hyper_galaxy = optimizer.variable.hyper_galaxy
            getattr(results_copy.constant, name).hyper_galaxy = optimizer.constant.hyper_galaxy

        return results_copy
