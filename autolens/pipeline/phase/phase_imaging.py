import copy

import numpy as np
from astropy import cosmology as cosmo

from autofit import conf
from autofit.mapper import prior as p
from autofit.mapper.model import ModelInstance
from autofit.optimize import non_linear
from autofit.tools.phase_property import PhaseProperty
from autolens import exc
from autolens.data.plotters import ccd_plotters
from autolens.lens import ray_tracing, lens_data as ld, lens_fit, sensitivity_fit
from autolens.lens.plotters import ray_tracing_plotters, lens_fit_plotters, \
    sensitivity_fit_plotters
from autolens.model.galaxy import galaxy as g
from autolens.model.inversion import pixelizations as px
from autolens.model.inversion import regularization as rg
from autolens.pipeline import tagging as tag
from autolens.pipeline.phase import Phase
from autolens.pipeline.phase.phase import Phase, setup_phase_mask


class PhaseImaging(Phase):

    def __init__(self, phase_name, tag_phases=True, phase_folders=None,
                 optimizer_class=non_linear.MultiNest,
                 sub_grid_size=2, bin_up_factor=None, image_psf_shape=None,
                 inversion_psf_shape=None, positions_threshold=None, mask_function=None,
                 inner_mask_radii=None,
                 interp_pixel_scale=None, cosmology=cosmo.Planck15,
                 auto_link_priors=False):

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
                                           tag_phases=tag_phases, optimizer_class=optimizer_class,
                                           cosmology=cosmology, auto_link_priors=auto_link_priors)

        self.sub_grid_size = sub_grid_size
        self.bin_up_factor = bin_up_factor
        self.image_psf_shape = image_psf_shape
        self.inversion_psf_shape = inversion_psf_shape
        self.positions_threshold = positions_threshold
        self.mask_function = mask_function
        self.inner_mask_radii = inner_mask_radii
        self.interp_pixel_scale = interp_pixel_scale

    @property
    def uses_inversion(self) -> bool:
        return False

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
        analysis = self.make_analysis(data=data, results=results, mask=mask,
                                      positions=positions)

        self.pass_priors(results)
        self.assert_and_save_pickle()

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
            positions = list(
                map(lambda position_set: np.asarray(position_set), positions))
        elif self.positions_threshold is None:
            positions = None
        elif self.positions_threshold is not None and positions is None:
            raise exc.PhaseException(
                'You have specified for a phase to use positions, but not input positions to the '
                'pipeline when you ran it.')

        lens_data = ld.LensData(ccd_data=data, mask=mask,
                                sub_grid_size=self.sub_grid_size,
                                image_psf_shape=self.image_psf_shape,
                                positions=positions,
                                interp_pixel_scale=self.interp_pixel_scale,
                                uses_inversion=self.uses_inversion)

        modified_image = self.modify_image(image=lens_data.unmasked_image,
                                           results=results)
        lens_data = lens_data.new_lens_data_with_modified_image(
            modified_image=modified_image)

        if self.bin_up_factor is not None:
            lens_data = lens_data.new_lens_data_with_binned_up_ccd_data_and_mask(
                bin_up_factor=self.bin_up_factor)

        self.output_phase_info()

        analysis = self.__class__.Analysis(lens_data=lens_data,
                                           cosmology=self.cosmology,
                                           positions_threshold=self.positions_threshold,
                                           results=results)
        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(self.optimizer.phase_output_path, 'phase.info')

        with open(file_phase_info, 'w') as phase_info:
            phase_info.write('Optimizer = {} \n'.format(type(self.optimizer).__name__))
            phase_info.write('Sub-grid size = {} \n'.format(self.sub_grid_size))
            phase_info.write('Image PSF shape = {} \n'.format(self.image_psf_shape))
            phase_info.write(
                'Pixelization PSF shape = {} \n'.format(self.inversion_psf_shape))
            phase_info.write(
                'Positions Threshold = {} \n'.format(self.positions_threshold))
            phase_info.write('Cosmology = {} \n'.format(self.cosmology))
            phase_info.write('Auto Link Priors = {} \n'.format(self.auto_link_priors))

            phase_info.close()

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):

        def __init__(self, lens_data, cosmology, positions_threshold, results=None):

            super(PhaseImaging.Analysis, self).__init__(cosmology=cosmology,
                                                        results=results)

            self.lens_data = lens_data

            self.positions_threshold = positions_threshold

            self.should_plot_image_plane_pix = \
                conf.instance.general.get('output',
                                          'plot_image_plane_adaptive_pixelization_grid',
                                          bool)

            self.plot_data_as_subplot = \
                conf.instance.general.get('output', 'plot_data_as_subplot', bool)
            self.plot_data_image = \
                conf.instance.general.get('output', 'plot_data_image', bool)
            self.plot_data_noise_map = \
                conf.instance.general.get('output', 'plot_data_noise_map', bool)
            self.plot_data_psf = \
                conf.instance.general.get('output', 'plot_data_psf', bool)
            self.plot_data_signal_to_noise_map = \
                conf.instance.general.get('output', 'plot_data_signal_to_noise_map',bool)
            self.plot_data_absolute_signal_to_noise_map = \
                conf.instance.general.get('output','plot_data_absolute_signal_to_noise_map', bool)
            self.plot_data_potential_chi_squared_map = \
                conf.instance.general.get('output','plot_data_potential_chi_squared_map', bool)

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
                conf.instance.general.get('output', 'plot_lens_fit_lens_model_image',bool)
            self.plot_lens_fit_source_model_image = \
                conf.instance.general.get('output', 'plot_lens_fit_source_model_image',bool)
            self.plot_lens_fit_source_plane_image = \
                conf.instance.general.get('output', 'plot_lens_fit_source_plane_image',bool)
            self.plot_lens_fit_residual_map = \
                conf.instance.general.get('output', 'plot_lens_fit_residual_map', bool)
            self.plot_lens_fit_chi_squared_map = \
                conf.instance.general.get('output', 'plot_lens_fit_chi_squared_map', bool)
            self.plot_lens_fit_contribution_map = \
                conf.instance.general.get('output', 'plot_lens_fit_contribution_map', bool)

            if self.last_results is not None:

                image_1d_galaxy_dict = {}
                self.hyper_model_image_1d = np.zeros(lens_data.mask_1d.shape)

                for galaxy, galaxy_image in results.last.image_2d_dict.items():
                    image_1d_galaxy_dict[galaxy] = lens_data.array_1d_from_array_2d(array_2d=galaxy_image)
                    self.check_for_previously_masked_values(array=image_1d_galaxy_dict[galaxy])

                self.hyper_galaxy_image_1d_name_dict = {}

                for name, galaxy in results.last.name_galaxy_tuples:

                    self.hyper_galaxy_image_1d_name_dict[name] = image_1d_galaxy_dict[name]

                    self.hyper_model_image_1d += image_1d_galaxy_dict[name]

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
            self.check_positions_trace_within_threshold(instance=instance)
            instance = self.associate_images(instance=instance)
            tracer = self.tracer_for_instance(instance=instance)
            fit = self.fit_for_tracers(tracer=tracer, padded_tracer=None)
            return fit.figure_of_merit

        def check_for_previously_masked_values(self, array):
            if not np.all(array) != 0.0:
                raise exc.PhaseException(
                    'When mapping a 2D array to a 1D array using lens data, a value encountered was'
                    '0.0 and therefore masked in a previous phase.')

        def associate_images(self, instance: ModelInstance) -> ModelInstance:
            """
            Takes images from the last result, if there is one, and associates them with galaxies in this phase where
            full-path galaxy names match.

            If the galaxy collection has a different name then an association is not made.

            e.g.
            lens_galaxies.lens will match with:
                lens_galaxies.lens
            but not with:
                galaxies.lens
                lens_galaxies.source

            Parameters
            ----------
            instance
                A model instance with 0 or more galaxies in its tree

            Returns
            -------
            instance
               The input instance with images associated with galaxies where possible.
            """
            if self.last_results is not None:
                for name, galaxy in instance.name_instance_tuples_for_class(g.Galaxy):
                    if name in self.hyper_galaxy_image_1d_name_dict:
                        galaxy.hyper_model_image_1d = self.hyper_model_image_1d
                        galaxy.hyper_galaxy_image_1d = self.hyper_galaxy_image_1d_name_dict[name]
                        galaxy.hyper_minimum_value = 0.0
            return instance

        def visualize(self, instance, image_path, during_analysis):

            mask = self.lens_data.mask_2d if self.should_plot_mask else None
            positions = self.lens_data.positions if self.should_plot_positions else None

            ccd_plotters.plot_ccd_for_phase(
                ccd_data=self.lens_data.ccd_data, mask=mask, positions=positions,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask,
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
                zoom_around_mask=self.zoom_around_mask, positions=positions,
                units=self.plot_units,
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
                fit=fit, during_analysis=during_analysis,
                should_plot_mask=self.should_plot_mask,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask,
                positions=positions,
                should_plot_image_plane_pix=self.should_plot_image_plane_pix,
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
            return lens_fit.LensDataFit.for_data_and_tracer(lens_data=self.lens_data,
                                                            tracer=tracer,
                                                            padded_tracer=padded_tracer)

        def check_positions_trace_within_threshold(self, instance):

            if self.lens_data.positions is not None:

                tracer = ray_tracing.TracerImageSourcePlanesPositions(
                    lens_galaxies=instance.lens_galaxies,
                    image_plane_positions=self.lens_data.positions)
                fit = lens_fit.LensPositionFit(positions=tracer.source_plane.positions,
                                               noise_map=self.lens_data.pixel_scale)

                if not fit.maximum_separation_within_threshold(
                        self.positions_threshold):
                    raise exc.RayTracingException

        def map_to_1d(self, data):
            """Convenience method"""
            return self.lens_data.mask.map_2d_array_to_masked_1d_array(data)


class MultiPlanePhase(PhaseImaging):
    """
    Fit a simple source and lens system.
    """

    galaxies = PhaseProperty("galaxies")

    def __init__(self, phase_name, tag_phases=True, phase_folders=None, galaxies=None,
                 optimizer_class=non_linear.MultiNest,
                 sub_grid_size=2, bin_up_factor=None, image_psf_shape=None,
                 positions_threshold=None,
                 mask_function=None,
                 inner_mask_radii=None, cosmology=cosmo.Planck15,
                 auto_link_priors=False):
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
                                              auto_link_priors=auto_link_priors)
        self.galaxies = galaxies

    @property
    def uses_inversion(self):
        for galaxy in self.galaxies:
            if galaxy.pixelization is not None:
                return True
        return False

    class Analysis(PhaseImaging.Analysis):

        def figure_of_merit_for_fit(self, tracer):
            raise NotImplementedError()

        def __init__(self, lens_data, cosmology, positions_threshold, results=None):
            self.lens_data = lens_data
            super(MultiPlanePhase.Analysis, self).__init__(lens_data=lens_data,
                                                           cosmology=cosmology,
                                                           positions_threshold=positions_threshold,
                                                           results=results)

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerMultiPlanes(galaxies=instance.galaxies,
                                                 image_plane_grid_stack=self.lens_data.grid_stack,
                                                 border=self.lens_data.border,
                                                 cosmology=self.cosmology)

        def padded_tracer_for_instance(self, instance):
            return ray_tracing.TracerMultiPlanes(galaxies=instance.galaxies,
                                                 image_plane_grid_stack=self.lens_data.padded_grid_stack,
                                                 cosmology=self.cosmology)

        @classmethod
        def describe(cls, instance):
            return "\nRunning multi-plane for... \n\nGalaxies:\n{}\n\n".format(
                instance.galaxies)


class LensSourcePlanePhase(PhaseImaging):
    """
    Fit a simple source and lens system.
    """

    lens_galaxies = PhaseProperty("lens_galaxies")
    source_galaxies = PhaseProperty("source_galaxies")

    def __init__(self, phase_name, tag_phases=True, phase_folders=None,
                 lens_galaxies=None, source_galaxies=None,
                 optimizer_class=non_linear.MultiNest,
                 sub_grid_size=2, bin_up_factor=None, image_psf_shape=None,
                 positions_threshold=None,
                 mask_function=None,
                 interp_pixel_scale=None, inner_mask_radii=None,
                 cosmology=cosmo.Planck15,
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
                                                   auto_link_priors=auto_link_priors)
        self.lens_galaxies = lens_galaxies or []
        self.source_galaxies = source_galaxies or []

    @property
    def uses_inversion(self):
        for galaxy_model in self.lens_galaxies:
            if galaxy_model.pixelization is not None:
                return True

        for galaxy_model in self.source_galaxies:
            if galaxy_model.pixelization is not None:
                return True
        return False

    class Analysis(PhaseImaging.Analysis):
        def figure_of_merit_for_fit(self, tracer):
            raise NotImplementedError()

        def tracer_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(
                lens_galaxies=instance.lens_galaxies,
                source_galaxies=instance.source_galaxies,
                image_plane_grid_stack=self.lens_data.grid_stack,
                border=self.lens_data.border, cosmology=self.cosmology)

        def padded_tracer_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(
                lens_galaxies=instance.lens_galaxies,
                source_galaxies=instance.source_galaxies,
                image_plane_grid_stack=self.lens_data.padded_grid_stack,
                cosmology=self.cosmology)

        @classmethod
        def describe(cls, instance):
            return "\nRunning lens/source lens for... \n\nLens Galaxy:\n{}\n\nSource " \
                   "Galaxy:\n{}\n\n".format(
                instance.lens_galaxies, instance.source_galaxies)

    class Result(PhaseImaging.Result):
        @property
        def unmasked_lens_plane_model_image(self):
            return self.most_likely_fit.unmasked_blurred_image_plane_image_of_planes[0]

        @property
        def unmasked_source_plane_model_image(self):
            return self.most_likely_fit.unmasked_blurred_image_plane_image_of_planes[1]


class LensPlanePhase(PhaseImaging):
    """
    Fit only the lens galaxy light.
    """

    lens_galaxies = PhaseProperty("lens_galaxies")

    def __init__(self, phase_name, tag_phases=True, phase_folders=None,
                 lens_galaxies=None,
                 optimizer_class=non_linear.MultiNest,
                 sub_grid_size=2, bin_up_factor=None,
                 image_psf_shape=None, mask_function=None, inner_mask_radii=None,
                 cosmology=cosmo.Planck15,
                 auto_link_priors=False):

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
                                             auto_link_priors=auto_link_priors)
        self.lens_galaxies = lens_galaxies

    @property
    def uses_inversion(self):
        for galaxy_model in self.lens_galaxies:
            if galaxy_model.pixelization is not None:
                return True
        return False

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
            return "\nRunning lens lens for... \n\nLens Galaxy::\n{}\n\n".format(
                instance.lens_galaxies)

    class Result(PhaseImaging.Result):

        @property
        def unmasked_lens_plane_model_image(self):
            return self.most_likely_fit.unmasked_blurred_image_plane_image_of_planes[0]


class SensitivityPhase(PhaseImaging):
    lens_galaxies = PhaseProperty("lens_galaxies")
    source_galaxies = PhaseProperty("source_galaxies")
    sensitive_galaxies = PhaseProperty("sensitive_galaxies")

    def __init__(self, phase_name, tag_phases=None, phase_folders=None,
                 lens_galaxies=None, source_galaxies=None,
                 sensitive_galaxies=None,
                 optimizer_class=non_linear.MultiNest, sub_grid_size=2,
                 bin_up_factor=None, mask_function=None,
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

        super(SensitivityPhase, self).__init__(phase_name=phase_name,
                                               tag_phases=tag_phases,
                                               phase_folders=phase_folders,
                                               optimizer_class=optimizer_class,
                                               sub_grid_size=sub_grid_size,
                                               bin_up_factor=bin_up_factor,
                                               mask_function=mask_function,
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
            fit = self.fit_for_tracers(tracer_normal=tracer_normal,
                                       tracer_sensitive=tracer_sensitive)
            return fit.figure_of_merit

        def visualize(self, instance, image_path, during_analysis):
            self.plot_count += 1

            tracer_normal = self.tracer_normal_for_instance(instance)
            tracer_sensitive = self.tracer_sensitive_for_instance(instance)
            fit = self.fit_for_tracers(tracer_normal=tracer_normal,
                                       tracer_sensitive=tracer_sensitive)

            ccd_plotters.plot_ccd_subplot(ccd_data=self.lens_data.ccd_data,
                                          mask=self.lens_data.mask,
                                          positions=self.lens_data.positions,
                                          output_path=image_path, output_format='png')

            ccd_plotters.plot_ccd_individual(ccd_data=self.lens_data.ccd_data,
                                             mask=self.lens_data.mask,
                                             positions=self.lens_data.positions,
                                             output_path=image_path,
                                             output_format='png')

            ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer_normal,
                                                          output_path=image_path,
                                                          output_format='png',
                                                          output_filename='tracer_normal')

            ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer_sensitive,
                                                          output_path=image_path,
                                                          output_format='png',
                                                          output_filename='tracer_sensitive')

            sensitivity_fit_plotters.plot_fit_subplot(fit=fit, output_path=image_path,
                                                      output_format='png')

            return fit

        def tracer_normal_for_instance(self, instance):
            return ray_tracing.TracerImageSourcePlanes(
                lens_galaxies=instance.lens_galaxies,
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
            return sensitivity_fit.fit_lens_data_with_sensitivity_tracers(
                lens_data=self.lens_data,
                tracer_normal=tracer_normal,
                tracer_sensitive=tracer_sensitive)

        @classmethod
        def describe(cls, instance):
            return "\nRunning lens/source lens for... \n\nLens Galaxy:\n{}\n\nSource " \
                   "Galaxy:\n{}\n\n Sensitive " \
                   "Galaxy\n{}\n\n ".format(instance.lens_galaxies,
                                            instance.source_galaxies,
                                            instance.sensitive_galaxies)


class PixelizationPhase(PhaseImaging):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def run(self, data, results=None, mask=None, positions=None):
        """
        Run the phase, overriding the optimizer's variable instance with one created to
        only fit pixelization hyperparameters.
        """
        variable = copy.deepcopy(results.last.variable)
        PixelizationPhase.transfer_classes(results.last.constant,variable)
        self.optimizer.variable = variable
        return super().run(data, results=results, mask=mask, positions=positions)

    @staticmethod
    def transfer_classes(instance, mapper):
        """
        Recursively overwrite priors in the mapper with constant values from the
        instance except where the containing class is associated with pixelization.

        Parameters
        ----------
        instance
            The best fit from the previous phase
        mapper
            The prior variable from the previous phase
        """
        for key, instance_value in instance.__dict__.items():
            try:
                mapper_value = getattr(mapper, key)
                if isinstance(mapper_value, p.Prior):
                    setattr(mapper, key, instance_value)
                if not (isinstance(instance_value, px.Pixelization) or isinstance(instance_value, rg.Regularization)):
                    try:
                        PixelizationPhase.transfer_classes(instance_value,mapper_value)
                    except AttributeError:
                        setattr(mapper, key, instance_value)
            except AttributeError:
                pass


class HyperGalaxyPhase(PhaseImaging):

    class Analysis(non_linear.Analysis):

        def __init__(self, lens_data, model_image_2d, galaxy_image_2d):
            """
            An analysis to fit the noise for a single galaxy image.

            Parameters
            ----------
            lens_data: LensData
                Lens data, including an image and noise
            model_image_2d: ndarray
                An image produce of the overall system by a model
            galaxy_image_2d: ndarray
                The contribution of one galaxy to the model image
            """
            self.lens_data = lens_data
            self.hyper_model_image_1d = lens_data.array_1d_from_array_2d(array_2d=model_image_2d)
            self.hyper_galaxy_image_1d = lens_data.array_1d_from_array_2d(array_2d=galaxy_image_2d)

            self.check_for_previously_masked_values(array=self.hyper_model_image_1d)
            self.check_for_previously_masked_values(array=self.hyper_galaxy_image_1d)

        def check_for_previously_masked_values(self, array):
            if not np.all(array) != 0.0:
                raise exc.PhaseException(
                    'When mapping a 2D array to a 1D array using lens data, a value encountered was'
                    '0.0 and therefore masked in a previous phase.')

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

            hyper_noise_1d = hyper_galaxy.hyper_noise_map_from_hyper_images_and_noise_map(
                hyper_model_image=self.hyper_model_image_1d, hyper_galaxy_image=self.hyper_galaxy_image_1d,
                noise_map=self.lens_data.noise_map_1d, hyper_minimum_value=0.0)

            hyper_noise_map_1d = self.lens_data.noise_map_1d + hyper_noise_1d
            return lens_fit.LensDataFit(image_1d=self.lens_data.image_1d, noise_map_1d=hyper_noise_map_1d,
                                        mask_1d=self.lens_data.mask_1d, model_image_1d=self.hyper_model_image_1d,
                                        map_to_scaled_array=self.lens_data.map_to_scaled_array)

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

        mask = setup_phase_mask(data=data, mask=mask, mask_function=self.mask_function,
                                inner_mask_radii=self.inner_mask_radii)

        lens_data = ld.LensData(ccd_data=data, mask=mask, sub_grid_size=self.sub_grid_size,
                                image_psf_shape=self.image_psf_shape, positions=positions,
                                interp_pixel_scale=self.interp_pixel_scale, uses_inversion=self.uses_inversion)

        model_image_2d = results.last.most_likely_fit.model_image_2d

        results_copy = copy.copy(results.last)

        for name, galaxy in results.last.name_galaxy_tuples:

            optimizer = self.optimizer.copy_with_name_extension(extension=name)
            optimizer.variable.hyper_galaxy = g.HyperGalaxy
            galaxy_image_2d = results.last.image_2d_dict[name]

            # If array is all zeros, galaxy did not have image in previous phase and should be ignored
            if not np.all(galaxy_image_2d==0):

                analysis = self.__class__.Analysis(lens_data=lens_data, model_image_2d=model_image_2d,
                                                    galaxy_image_2d=galaxy_image_2d)
                optimizer.fit(analysis)

                getattr(results_copy.variable, name).hyper_galaxy = optimizer.variable.hyper_galaxy
                getattr(results_copy.constant, name).hyper_galaxy = optimizer.constant.hyper_galaxy

        return results_copy