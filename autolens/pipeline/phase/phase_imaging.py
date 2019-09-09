import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from autolens import exc
from autolens.lens import ray_tracing, lens_data as ld, lens_fit
from autolens.model.galaxy import galaxy as g
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.pipeline import phase_tagging
from autolens.pipeline.phase import phase_extensions
from autolens.pipeline.phase.phase import Phase, setup_phase_mask
from autolens.pipeline.plotters import phase_plotters


def isinstance_or_prior(obj, cls):
    if isinstance(obj, cls):
        return True
    if isinstance(obj, af.PriorModel) and obj.cls == cls:
        return True
    return False


class PhaseImaging(Phase):
    hyper_image_sky = af.PhaseProperty("hyper_image_sky")
    hyper_background_noise = af.PhaseProperty("hyper_background_noise")
    galaxies = af.PhaseProperty("galaxies")

    def __init__(
        self,
        phase_name,
        phase_folders=tuple(),
        galaxies=None,
        hyper_image_sky=None,
        hyper_background_noise=None,
        optimizer_class=af.MultiNest,
        cosmology=cosmo.Planck15,
        sub_grid_size=2,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        psf_shape=None,
        positions_threshold=None,
        mask_function=None,
        inner_mask_radii=None,
        pixel_scale_interpolation_grid=None,
        pixel_scale_binned_cluster_grid=None,
        inversion_uses_border=True,
        inversion_pixel_limit=None,
        auto_link_priors=False,
    ):

        """

        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        pixel_scale_binned_cluster_grid : float or None
            If *True*, the hyper_galaxies image used to generate the cluster'grids weight map will be binned up to this \
            higher pixel scale to speed up the KMeans clustering algorithm.
        """

        phase_tag = phase_tagging.phase_tag_from_phase_settings(
            sub_grid_size=sub_grid_size,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            psf_shape=psf_shape,
            positions_threshold=positions_threshold,
            inner_mask_radii=inner_mask_radii,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            pixel_scale_binned_cluster_grid=pixel_scale_binned_cluster_grid,
        )

        super(PhaseImaging, self).__init__(
            phase_name=phase_name,
            phase_tag=phase_tag,
            phase_folders=phase_folders,
            optimizer_class=optimizer_class,
            cosmology=cosmology,
            auto_link_priors=auto_link_priors,
        )

        self.sub_grid_size = sub_grid_size
        self.signal_to_noise_limit = signal_to_noise_limit
        self.bin_up_factor = bin_up_factor
        self.psf_shape = psf_shape
        self.positions_threshold = positions_threshold
        self.mask_function = mask_function
        self.inner_mask_radii = inner_mask_radii
        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid
        self.pixel_scale_binned_cluster_grid = pixel_scale_binned_cluster_grid
        self.inversion_uses_border = inversion_uses_border

        if inversion_pixel_limit is not None:
            self.inversion_pixel_limit = inversion_pixel_limit
        else:
            self.inversion_pixel_limit = af.conf.instance.general.get(
                "inversion", "inversion_pixel_limit_overall", int
            )

        self.galaxies = galaxies or []
        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.hyper_noise_map_max = af.conf.instance.general.get(
            "hyper", "hyper_noise_map_max", float
        )

        self.is_hyper_phase = False

    @property
    def pixelization(self):
        for galaxy in self.galaxies:
            if galaxy.pixelization is not None:
                if isinstance(galaxy.pixelization, af.PriorModel):
                    return galaxy.pixelization.cls
                else:
                    return galaxy.pixelization

    @property
    def uses_cluster_inversion(self):
        if self.galaxies:
            for galaxy in self.galaxies:
                if isinstance_or_prior(galaxy.pixelization, pix.VoronoiBrightnessImage):
                    return True

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
            A result object comprising the best fit model and other hyper_galaxies.
        """
        analysis = self.make_analysis(
            data=data, results=results, mask=mask, positions=positions
        )

        self.variable = self.variable.populate(results)
        self.customize_priors(results)
        self.assert_and_save_pickle()

        result = self.run_analysis(analysis)

        return self.make_result(result=result, analysis=analysis)

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

        mask = setup_phase_mask(
            data=data,
            mask=mask,
            mask_function=self.mask_function,
            inner_mask_radii=self.inner_mask_radii,
        )

        if self.positions_threshold is not None and positions is None:
            raise exc.PhaseException(
                "You have specified for a phase to use positions, but not input positions to the "
                "pipeline when you ran it."
            )

        pixel_scale_binned_grid = None

        if self.uses_cluster_inversion:

            if self.pixel_scale_binned_cluster_grid is None:

                pixel_scale_binned_cluster_grid = mask.pixel_scale

            else:

                pixel_scale_binned_cluster_grid = self.pixel_scale_binned_cluster_grid

            if pixel_scale_binned_cluster_grid > mask.pixel_scale:

                bin_up_factor = int(
                    self.pixel_scale_binned_cluster_grid / mask.pixel_scale
                )

            else:

                bin_up_factor = 1

            binned_mask = mask.binned_up_mask_from_mask(bin_up_factor=bin_up_factor)

            while binned_mask.pixels_in_mask < self.inversion_pixel_limit:

                if bin_up_factor == 1:
                    raise exc.DataException(
                        "The pixelization "
                        + str(self.pixelization)
                        + " uses a KMeans clustering algorithm which uses"
                        "a hyper model image to adapt the pixelization. This hyper model image must have more pixels"
                        "than inversion pixels. Current, the inversion_pixel_limit exceeds the data-points in the image.\n\n"
                        "To rectify this image, manually set the inversion pixel limit in the pipeline phases or change the inversion_pixel_limit_overall parameter in general.ini"
                    )

                bin_up_factor -= 1
                binned_mask = mask.binned_up_mask_from_mask(bin_up_factor=bin_up_factor)

            pixel_scale_binned_grid = mask.pixel_scale * bin_up_factor

        preload_pixelization_grids_of_planes = None

        if results is not None:
            if results.last is not None:
                if hasattr(results.last, "hyper_combined"):
                    if self.pixelization is not None:
                        if type(self.pixelization) == type(results.last.pixelization):
                            preload_pixelization_grids_of_planes = (
                                results.last.hyper_combined.most_likely_pixelization_grids_of_planes
                            )

        if self.is_hyper_phase:
            preload_pixelization_grids_of_planes = None

        lens_data = ld.LensData(
            ccd_data=data,
            mask=mask,
            sub_grid_size=self.sub_grid_size,
            trimmed_psf_shape=self.psf_shape,
            positions=positions,
            positions_threshold=self.positions_threshold,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=pixel_scale_binned_grid,
            hyper_noise_map_max=self.hyper_noise_map_max,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            preload_pixelization_grids_of_planes=preload_pixelization_grids_of_planes,
        )

        modified_image = self.modify_image(
            image=lens_data.unmasked_image, results=results
        )

        lens_data = lens_data.new_lens_data_with_modified_image(
            modified_image=modified_image
        )

        if self.signal_to_noise_limit is not None:
            lens_data = lens_data.new_lens_data_with_signal_to_noise_limit(
                signal_to_noise_limit=self.signal_to_noise_limit
            )

        if self.bin_up_factor is not None:
            lens_data = lens_data.new_lens_data_with_binned_up_ccd_data_and_mask(
                bin_up_factor=self.bin_up_factor
            )

        self.output_phase_info()

        analysis = self.Analysis(
            lens_data=lens_data,
            cosmology=self.cosmology,
            image_path=self.optimizer.image_path,
            results=results,
        )

        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(self.optimizer.phase_output_path, "phase.info")

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.optimizer).__name__))
            phase_info.write("Sub-grid size = {} \n".format(self.sub_grid_size))
            phase_info.write("PSF shape = {} \n".format(self.psf_shape))
            phase_info.write(
                "Positions Threshold = {} \n".format(self.positions_threshold)
            )
            phase_info.write("Cosmology = {} \n".format(self.cosmology))
            phase_info.write("Auto Link Priors = {} \n".format(self.auto_link_priors))

            phase_info.close()

    def extend_with_inversion_phase(self):
        return phase_extensions.InversionPhase(phase=self)

    def extend_with_multiple_hyper_phases(
        self,
        hyper_galaxy=False,
        inversion=False,
        include_background_sky=False,
        include_background_noise=False,
    ):

        if hyper_galaxy:
            if not include_background_sky and not include_background_noise:
                phase_hyper_galaxy = (
                    phase_extensions.hyper_galaxy_phase.HyperGalaxyPhase
                )
            elif include_background_sky and not include_background_noise:
                phase_hyper_galaxy = (
                    phase_extensions.hyper_galaxy_phase.HyperGalaxyBackgroundSkyPhase
                )
            elif not include_background_sky and include_background_noise:
                phase_hyper_galaxy = (
                    phase_extensions.hyper_galaxy_phase.HyperGalaxyBackgroundNoisePhase
                )
            else:
                phase_hyper_galaxy = (
                    phase_extensions.hyper_galaxy_phase.HyperGalaxyBackgroundBothPhase
                )
        else:
            phase_hyper_galaxy = None

        if inversion:
            if not include_background_sky and not include_background_noise:
                phase_inversion = phase_extensions.InversionPhase
            elif include_background_sky and not include_background_noise:
                phase_inversion = phase_extensions.InversionBackgroundSkyPhase
            elif not include_background_sky and include_background_noise:
                phase_inversion = phase_extensions.InversionBackgroundNoisePhase
            else:
                phase_inversion = phase_extensions.InversionBackgroundBothPhase
        else:
            phase_inversion = None

        hyper_phase_classes = tuple(filter(None, (phase_hyper_galaxy, phase_inversion)))

        if len(hyper_phase_classes) == 0:
            return self
        else:
            return phase_extensions.CombinedHyperPhase(
                phase=self, hyper_phase_classes=hyper_phase_classes
            )

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):
        def __init__(self, lens_data, cosmology, image_path=None, results=None):

            super(PhaseImaging.Analysis, self).__init__(
                cosmology=cosmology, results=results
            )

            self.lens_data = lens_data

            self.should_plot_image_plane_pix = af.conf.instance.visualize.get(
                "figures", "plot_image_plane_adaptive_pixelization_grid", bool
            )

            self.plot_data_as_subplot = af.conf.instance.visualize.get(
                "plots", "plot_data_as_subplot", bool
            )

            self.plot_data_image = af.conf.instance.visualize.get(
                "plots", "plot_data_image", bool
            )

            self.plot_data_noise_map = af.conf.instance.visualize.get(
                "plots", "plot_data_noise_map", bool
            )

            self.plot_data_psf = af.conf.instance.visualize.get(
                "plots", "plot_data_psf", bool
            )

            self.plot_data_signal_to_noise_map = af.conf.instance.visualize.get(
                "plots", "plot_data_signal_to_noise_map", bool
            )

            self.plot_data_absolute_signal_to_noise_map = af.conf.instance.visualize.get(
                "plots", "plot_data_absolute_signal_to_noise_map", bool
            )

            self.plot_data_potential_chi_squared_map = af.conf.instance.visualize.get(
                "plots", "plot_data_potential_chi_squared_map", bool
            )

            self.plot_lens_fit_all_at_end_png = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_all_at_end_png", bool
            )
            self.plot_lens_fit_all_at_end_fits = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_all_at_end_fits", bool
            )

            self.plot_lens_fit_as_subplot = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_as_subplot", bool
            )

            self.plot_lens_fit_of_planes_as_subplot = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_of_planes_as_subplot", bool
            )

            self.plot_lens_fit_inversion_as_subplot = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_inversion_as_subplot", bool
            )

            self.plot_lens_fit_image = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_image", bool
            )

            self.plot_lens_fit_noise_map = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_noise_map", bool
            )

            self.plot_lens_fit_signal_to_noise_map = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_signal_to_noise_map", bool
            )

            self.plot_lens_fit_model_image = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_model_image", bool
            )

            self.plot_lens_fit_residual_map = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_residual_map", bool
            )

            self.plot_lens_fit_normalized_residual_map = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_normalized_residual_map", bool
            )

            self.plot_lens_fit_chi_squared_map = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_chi_squared_map", bool
            )

            self.plot_lens_fit_contribution_maps = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_contribution_maps", bool
            )

            self.plot_lens_fit_pixelization_residual_map = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_pixelization_residual_map", bool
            )

            self.plot_lens_fit_pixelization_normalized_residuals = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_pixelization_normalized_residual_map", bool
            )

            self.plot_lens_fit_pixelization_chi_squared_map = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_pixelization_chi_squared_map", bool
            )

            self.plot_lens_fit_pixelization_regularization_weights = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_pixelization_regularization_weight_map", bool
            )

            self.plot_lens_fit_subtracted_images_of_planes = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_subtracted_images_of_planes", bool
            )

            self.plot_lens_fit_model_images_of_planes = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_model_images_of_planes", bool
            )

            self.plot_lens_fit_plane_images_of_planes = af.conf.instance.visualize.get(
                "plots", "plot_lens_fit_plane_images_of_planes", bool
            )

            mask = self.lens_data.mask_2d if self.should_plot_mask else None
            positions = self.lens_data.positions if self.should_plot_positions else None

            subplot_path = af.path_util.make_and_return_path_from_path_and_folder_names(
                path=image_path, folder_names=["subplots"]
            )

            phase_plotters.plot_ccd_for_phase(
                ccd_data=self.lens_data.ccd_data,
                mask=mask,
                positions=positions,
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
                visualize_path=image_path,
                subplot_path=subplot_path,
            )

            self.plot_hyper_model_image = af.conf.instance.visualize.get(
                "plots", "plot_hyper_model_image", bool
            )

            self.plot_hyper_galaxy_images = af.conf.instance.visualize.get(
                "plots", "plot_hyper_galaxy_images", bool
            )

            self.plot_binned_hyper_galaxy_images = af.conf.instance.visualize.get(
                "plots", "plot_binned_hyper_galaxy_images", bool
            )

            if self.last_results is not None:

                self.hyper_galaxy_image_1d_path_dict = (
                    self.last_results.hyper_galaxy_image_1d_path_dict
                )

                self.hyper_model_image_1d = self.last_results.hyper_model_image_1d

                self.binned_hyper_galaxy_image_1d_path_dict = self.last_results.binned_hyper_galaxy_image_1d_path_dict_from_binned_grid(
                    binned_grid=lens_data.grid.binned
                )

                if mask is not None:
                    phase_plotters.plot_hyper_images_for_phase(
                        hyper_model_image_2d=mask.scaled_array_2d_from_array_1d(
                            array_1d=self.hyper_model_image_1d
                        ),
                        hyper_galaxy_image_2d_path_dict=self.last_results.hyper_galaxy_image_2d_path_dict,
                        binned_hyper_galaxy_image_2d_path_dict=self.last_results.binned_hyper_galaxy_image_2d_path_dict_from_binned_grid(
                            binned_grid=lens_data.grid.binned
                        ),
                        mask=lens_data.mask_2d,
                        binned_grid=lens_data.grid.binned,
                        extract_array_from_mask=self.extract_array_from_mask,
                        zoom_around_mask=self.zoom_around_mask,
                        units=self.plot_units,
                        should_plot_hyper_model_image=self.plot_hyper_model_image,
                        should_plot_hyper_galaxy_images=self.plot_hyper_galaxy_images,
                        should_plot_binned_hyper_galaxy_images=self.plot_binned_hyper_galaxy_images,
                        visualize_path=image_path,
                    )

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

            tracer = self.tracer_for_instance(instance=instance)

            self.check_positions_trace_within_threshold_via_tracer(tracer=tracer)
            self.check_inversion_pixels_are_below_limit_via_tracer(tracer=tracer)

            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_background_noise = self.hyper_background_noise_for_instance(
                instance=instance
            )

            fit = self.fit_for_tracer(
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            return fit.figure_of_merit

        def associate_images(self, instance: af.ModelInstance) -> af.ModelInstance:
            """
            Takes images from the last result, if there is one, and associates them with galaxies in this phase where
            full-path galaxy names match.

            If the galaxy collection has a different name then an association is not made.

            e.g.
            galaxies.lens will match with:
                galaxies.lens
            but not with:
                galaxies.lens
                galaxies.source

            Parameters
            ----------
            instance
                A model instance with 0 or more galaxies in its tree

            Returns
            -------
            instance
               The input instance with images associated with galaxies where possible.
            """
            if hasattr(self, "hyper_galaxy_image_1d_path_dict"):
                for galaxy_path, galaxy in instance.path_instance_tuples_for_class(
                    g.Galaxy
                ):
                    if galaxy_path in self.hyper_galaxy_image_1d_path_dict:
                        galaxy.hyper_model_image_1d = self.hyper_model_image_1d
                        galaxy.hyper_galaxy_image_1d = self.hyper_galaxy_image_1d_path_dict[
                            galaxy_path
                        ]
                        if (
                            hasattr(self, "binned_hyper_galaxy_image_1d_path_dict")
                            and self.binned_hyper_galaxy_image_1d_path_dict is not None
                        ):
                            galaxy.binned_hyper_galaxy_image_1d = self.binned_hyper_galaxy_image_1d_path_dict[
                                galaxy_path
                            ]
            return instance

        def hyper_image_sky_for_instance(self, instance):

            if hasattr(instance, "hyper_image_sky"):
                return instance.hyper_image_sky
            else:
                return None

        def hyper_background_noise_for_instance(self, instance):

            if hasattr(instance, "hyper_background_noise"):
                return instance.hyper_background_noise
            else:
                return None

        def tracer_for_instance(self, instance):

            instance = self.associate_images(instance=instance)

            return ray_tracing.Tracer.from_galaxies(
                galaxies=instance.galaxies, cosmology=self.cosmology
            )

        def fit_for_tracer(self, tracer, hyper_image_sky, hyper_background_noise):

            return lens_fit.LensDataFit.for_data_and_tracer(
                lens_data=self.lens_data,
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

        def check_positions_trace_within_threshold_via_tracer(self, tracer):

            if (
                self.lens_data.positions is not None
                and self.lens_data.positions_threshold is not None
            ):

                traced_positions_of_planes = tracer.traced_positions_of_planes_from_positions(
                    positions=self.lens_data.positions
                )

                fit = lens_fit.LensPositionFit(
                    positions=traced_positions_of_planes[-1],
                    noise_map=self.lens_data.pixel_scale,
                )

                if not fit.maximum_separation_within_threshold(
                    self.lens_data.positions_threshold
                ):
                    raise exc.RayTracingException

        def check_inversion_pixels_are_below_limit_via_tracer(self, tracer):

            if self.lens_data.inversion_pixel_limit is not None:
                pixelizations = list(filter(None, tracer.pixelizations_of_planes))
                if pixelizations:
                    for pixelization in pixelizations:
                        if pixelization.pixels > self.lens_data.inversion_pixel_limit:
                            raise exc.PixelizationException

        def map_to_1d(self, data):
            """Convenience method"""
            return self.lens_data.mask.array_1d_from_array_2d(data)

        @classmethod
        def describe(cls, instance):
            return "\nRunning for... \n\nGalaxies:\n{}\n\n".format(instance.galaxies)

        def visualize(self, instance, image_path, during_analysis):

            subplot_path = af.path_util.make_and_return_path_from_path_and_folder_names(
                path=image_path, folder_names=["subplots"]
            )

            instance = self.associate_images(instance=instance)

            mask = self.lens_data.mask_2d if self.should_plot_mask else None
            positions = self.lens_data.positions if self.should_plot_positions else None

            tracer = self.tracer_for_instance(instance=instance)

            phase_plotters.plot_ray_tracing_for_phase(
                tracer=tracer,
                grid=self.lens_data.grid,
                during_analysis=during_analysis,
                mask=mask,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask,
                positions=positions,
                units=self.plot_units,
                should_plot_as_subplot=self.plot_ray_tracing_as_subplot,
                should_plot_all_at_end_png=self.plot_ray_tracing_all_at_end_png,
                should_plot_all_at_end_fits=self.plot_ray_tracing_all_at_end_fits,
                should_plot_image_plane_image=self.plot_ray_tracing_profile_image,
                should_plot_source_plane=self.plot_ray_tracing_source_plane,
                should_plot_convergence=self.plot_ray_tracing_convergence,
                should_plot_potential=self.plot_ray_tracing_potential,
                should_plot_deflections=self.plot_ray_tracing_deflections,
                visualize_path=image_path,
                subplot_path=subplot_path,
            )

            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_background_noise = self.hyper_background_noise_for_instance(
                instance=instance
            )

            fit = self.fit_for_tracer(
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            phase_plotters.plot_lens_fit_for_phase(
                fit=fit,
                during_analysis=during_analysis,
                should_plot_mask=self.should_plot_mask,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask,
                positions=positions,
                should_plot_image_plane_pix=self.should_plot_image_plane_pix,
                should_plot_all_at_end_png=self.plot_lens_fit_all_at_end_png,
                should_plot_all_at_end_fits=self.plot_lens_fit_all_at_end_fits,
                should_plot_fit_as_subplot=self.plot_lens_fit_as_subplot,
                should_plot_fit_of_planes_as_subplot=self.plot_lens_fit_of_planes_as_subplot,
                should_plot_inversion_as_subplot=self.plot_lens_fit_inversion_as_subplot,
                should_plot_image=self.plot_lens_fit_image,
                should_plot_noise_map=self.plot_lens_fit_noise_map,
                should_plot_signal_to_noise_map=self.plot_lens_fit_signal_to_noise_map,
                should_plot_model_image=self.plot_lens_fit_model_image,
                should_plot_residual_map=self.plot_lens_fit_residual_map,
                should_plot_normalized_residual_map=self.plot_lens_fit_normalized_residual_map,
                should_plot_chi_squared_map=self.plot_lens_fit_chi_squared_map,
                should_plot_pixelization_residual_map=self.plot_lens_fit_pixelization_residual_map,
                should_plot_pixelization_normalized_residual_map=self.plot_lens_fit_normalized_residual_map,
                should_plot_pixelization_chi_squared_map=self.plot_lens_fit_pixelization_chi_squared_map,
                should_plot_pixelization_regularization_weights=self.plot_lens_fit_pixelization_regularization_weights,
                should_plot_subtracted_images_of_planes=self.plot_lens_fit_subtracted_images_of_planes,
                should_plot_model_images_of_planes=self.plot_lens_fit_model_images_of_planes,
                should_plot_plane_images_of_planes=self.plot_lens_fit_plane_images_of_planes,
                units=self.plot_units,
                visualize_path=image_path,
                subplot_path=subplot_path,
            )
