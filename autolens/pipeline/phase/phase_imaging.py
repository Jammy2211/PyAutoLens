import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from autolens import exc
from autolens.lens import ray_tracing, lens_data as ld, lens_fit
from autolens.model.galaxy import galaxy as g
from autolens.pipeline import phase_tagging
from autolens.pipeline.phase import phase_extensions
from autolens.pipeline.phase.phase import Phase, setup_phase_mask
from autolens.pipeline.plotters import phase_plotters


class PhaseImaging(Phase):

    hyper_image_sky = af.PhaseProperty("hyper_image_sky")
    hyper_noise_background = af.PhaseProperty("hyper_noise_background")
    galaxies = af.PhaseProperty("galaxies")

    def __init__(
        self,
        phase_name,
        tag_phases=True,
        phase_folders=tuple(),
        galaxies=None,
        hyper_image_sky=None,
        hyper_noise_background=None,
        optimizer_class=af.MultiNest,
        sub_grid_size=2,
        bin_up_factor=None,
        image_psf_shape=None,
        inversion_psf_shape=None,
        positions_threshold=None,
        mask_function=None,
        inner_mask_radii=None,
        interp_pixel_scale=None,
        use_inversion_border=True,
        inversion_pixel_limit=None,
        cluster_pixel_scale=None,
        cosmology=cosmo.Planck15,
        auto_link_priors=False,
    ):

        """

        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        cluster_pixel_scale : float or None
            If *True*, the hyper image used to generate the cluster'grids weight map will be binned up to this \
            higher pixel scale to speed up the KMeans clustering algorithm.
        """

        if tag_phases:

            phase_tag = phase_tagging.phase_tag_from_phase_settings(
                sub_grid_size=sub_grid_size,
                bin_up_factor=bin_up_factor,
                image_psf_shape=image_psf_shape,
                inversion_psf_shape=inversion_psf_shape,
                positions_threshold=positions_threshold,
                inner_mask_radii=inner_mask_radii,
                interp_pixel_scale=interp_pixel_scale,
                cluster_pixel_scale=cluster_pixel_scale,
            )

        else:

            phase_tag = None

        super(PhaseImaging, self).__init__(
            phase_name=phase_name,
            phase_tag=phase_tag,
            phase_folders=phase_folders,
            tag_phases=tag_phases,
            optimizer_class=optimizer_class,
            cosmology=cosmology,
            auto_link_priors=auto_link_priors,
        )

        self.sub_grid_size = sub_grid_size
        self.bin_up_factor = bin_up_factor
        self.image_psf_shape = image_psf_shape
        self.inversion_psf_shape = inversion_psf_shape
        self.positions_threshold = positions_threshold
        self.mask_function = mask_function
        self.inner_mask_radii = inner_mask_radii
        self.interp_pixel_scale = interp_pixel_scale
        self.use_inversion_border = use_inversion_border
        self.inversion_pixel_limit = inversion_pixel_limit
        self.cluster_pixel_scale = cluster_pixel_scale

        inversion_pixel_limit_from_prior = int(
            af.conf.instance.prior_default.get(
                "pixelizations", "VoronoiBrightnessImage", "pixels"
            )[2]
        )

        if self.inversion_pixel_limit is not None:

            self.cluster_pixel_limit = min(
                inversion_pixel_limit_from_prior, self.inversion_pixel_limit
            )

        else:

            self.cluster_pixel_limit = inversion_pixel_limit_from_prior
            self.inversion_pixel_limit = self.cluster_pixel_limit

        self.galaxies = galaxies or []
        self.hyper_image_sky = hyper_image_sky
        self.hyper_noise_background = hyper_noise_background

    @property
    def uses_hyper_images(self):
        if self.galaxies:
            return any([galaxy.uses_hyper_images for galaxy in self.galaxies])
        else:
            return False

    @property
    def uses_inversion(self):
        if self.galaxies:
            for galaxy in self.galaxies:
                if galaxy.uses_inversion:
                    return True
        return False

    @property
    def uses_cluster_inversion(self):
        if self.galaxies:
            for galaxy in self.galaxies:
                if galaxy.uses_cluster_inversion:
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
            A result object comprising the best fit model and other hyper.
        """
        analysis = self.make_analysis(
            data=data, results=results, mask=mask, positions=positions
        )

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

        mask = setup_phase_mask(
            data=data,
            mask=mask,
            mask_function=self.mask_function,
            inner_mask_radii=self.inner_mask_radii,
        )

        if self.positions_threshold is not None and positions is not None:
            positions = list(
                map(lambda position_set: np.asarray(position_set), positions)
            )
        elif self.positions_threshold is None:
            positions = None
        elif self.positions_threshold is not None and positions is None:
            raise exc.PhaseException(
                "You have specified for a phase to use positions, but not input positions to the "
                "pipeline when you ran it."
            )

        lens_data = ld.LensData(
            ccd_data=data,
            mask=mask,
            sub_grid_size=self.sub_grid_size,
            image_psf_shape=self.image_psf_shape,
            positions=positions,
            interp_pixel_scale=self.interp_pixel_scale,
            cluster_pixel_scale=self.cluster_pixel_scale,
            cluster_pixel_limit=self.cluster_pixel_limit,
            uses_inversion=self.uses_inversion,
            uses_cluster_inversion=self.uses_cluster_inversion,
        )

        modified_image = self.modify_image(
            image=lens_data.unmasked_image, results=results
        )

        lens_data = lens_data.new_lens_data_with_modified_image(
            modified_image=modified_image
        )

        if self.bin_up_factor is not None:
            lens_data = lens_data.new_lens_data_with_binned_up_ccd_data_and_mask(
                bin_up_factor=self.bin_up_factor
            )

        self.output_phase_info()

        analysis = self.Analysis(
            lens_data=lens_data,
            cosmology=self.cosmology,
            positions_threshold=self.positions_threshold,
            image_path=self.optimizer.image_path,
            results=results,
            use_inversion_border=self.use_inversion_border,
            inversion_pixel_limit=self.inversion_pixel_limit,
        )

        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(self.optimizer.phase_output_path, "phase.info")

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.optimizer).__name__))
            phase_info.write("Sub-grid size = {} \n".format(self.sub_grid_size))
            phase_info.write("Image PSF shape = {} \n".format(self.image_psf_shape))
            phase_info.write(
                "Pixelization PSF shape = {} \n".format(self.inversion_psf_shape)
            )
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

        phase_hyper_galaxy = None

        if hyper_galaxy:
            phase_hyper_galaxy = phase_extensions.HyperGalaxyPhase

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

        hyper_phase_classes = tuple(filter(None, (phase_inversion, phase_hyper_galaxy)))

        return phase_extensions.CombinedHyperPhase(
            phase=self, hyper_phase_classes=hyper_phase_classes
        )

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):
        def __init__(
            self,
            lens_data,
            cosmology,
            positions_threshold,
            use_inversion_border=True,
            inversion_pixel_limit=None,
            image_path=None,
            results=None,
        ):

            super(PhaseImaging.Analysis, self).__init__(
                cosmology=cosmology, results=results
            )

            self.lens_data = lens_data

            self.positions_threshold = positions_threshold

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

            self.use_inversion_border = use_inversion_border
            self.inversion_pixel_limit = inversion_pixel_limit

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

            self.plot_hyper_galaxy_cluster_images = af.conf.instance.visualize.get(
                "plots", "plot_hyper_galaxy_cluster_images", bool
            )

            self.preload_pixelization_grid = None

            if self.last_results is not None:

                self.hyper_galaxy_image_1d_path_dict = (
                    self.last_results.hyper_galaxy_image_1d_path_dict
                )

                self.hyper_model_image_1d = self.last_results.hyper_model_image_1d

                self.hyper_galaxy_cluster_image_1d_path_dict = self.last_results.hyper_galaxy_cluster_image_1d_path_dict_from_cluster(
                    cluster=lens_data.cluster
                )

                phase_plotters.plot_hyper_images_for_phase(
                    hyper_model_image_2d=mask.scaled_array_2d_from_array_1d(
                        array_1d=self.hyper_model_image_1d
                    ),
                    hyper_galaxy_image_2d_path_dict=self.last_results.hyper_galaxy_image_2d_path_dict,
                    hyper_galaxy_cluster_image_2d_path_dict=self.last_results.hyper_galaxy_cluster_image_2d_path_dict_from_cluster(
                        cluster=lens_data.cluster
                    ),
                    mask=lens_data.mask_2d,
                    cluster=lens_data.cluster,
                    extract_array_from_mask=self.extract_array_from_mask,
                    zoom_around_mask=self.zoom_around_mask,
                    units=self.plot_units,
                    should_plot_hyper_model_image=self.plot_hyper_model_image,
                    should_plot_hyper_galaxy_images=self.plot_hyper_galaxy_images,
                    should_plot_hyper_galaxy_cluster_images=self.plot_hyper_galaxy_cluster_images,
                    visualize_path=image_path,
                )

                if hasattr(self.results.last, "hyper_combined"):

                    self.preload_pixelization_grid = (
                        self.results.last.hyper_combined.most_likely_image_plane_pixelization_grid
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
            self.check_positions_trace_within_threshold(instance=instance)
            self.check_inversion_pixels_are_below_limit(instance=instance)
            tracer = self.tracer_for_instance(instance=instance)

            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_noise_background = self.hyper_noise_background_for_instance(
                instance=instance
            )

            fit = self.fit_for_tracer(
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_noise_background=hyper_noise_background,
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
                        if self.hyper_galaxy_cluster_image_1d_path_dict is not None:
                            galaxy.hyper_galaxy_cluster_image_1d = self.hyper_galaxy_cluster_image_1d_path_dict[
                                galaxy_path
                            ]
            return instance

        def add_grids_to_grid_stack(self, galaxies, grid_stack):

            if self.preload_pixelization_grid is None:

                for galaxy in galaxies:
                    if galaxy.pixelization is not None:
                        pixelization_grid = galaxy.pixelization.pixelization_grid_from_grid_stack(
                            grid_stack=grid_stack,
                            hyper_image=galaxy.hyper_galaxy_cluster_image_1d,
                            cluster=self.lens_data.cluster,
                            seed=1,
                        )

                        return grid_stack.new_grid_stack_with_grids_added(
                            pixelization=pixelization_grid
                        )

            else:

                return grid_stack.new_grid_stack_with_grids_added(
                    pixelization=self.preload_pixelization_grid
                )

            return grid_stack

        def hyper_image_sky_for_instance(self, instance):

            if hasattr(instance, "hyper_image_sky"):
                return instance.hyper_image_sky
            else:
                return None

        def hyper_noise_background_for_instance(self, instance):

            if hasattr(instance, "hyper_noise_background"):
                return instance.hyper_noise_background
            else:
                return None

        def tracer_for_instance(self, instance):

            instance = self.associate_images(instance=instance)

            image_plane_grid_stack = self.add_grids_to_grid_stack(
                galaxies=instance.galaxies, grid_stack=self.lens_data.grid_stack
            )

            if self.use_inversion_border:
                border = self.lens_data.border
            else:
                border = None

            return ray_tracing.Tracer.from_galaxies_and_image_plane_grid_stack(
                galaxies=instance.galaxies,
                image_plane_grid_stack=image_plane_grid_stack,
                border=border,
                cosmology=self.cosmology,
            )

        def fit_for_tracer(self, tracer, hyper_image_sky, hyper_noise_background):

            return lens_fit.LensDataFit.for_data_and_tracer(
                lens_data=self.lens_data,
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_noise_background=hyper_noise_background,
            )

        def check_positions_trace_within_threshold(self, instance):

            if self.lens_data.positions is not None:

                tracer = ray_tracing.Tracer.from_galaxies_and_image_plane_positions(
                    galaxies=instance.galaxies,
                    image_plane_positions=self.lens_data.positions,
                )
                fit = lens_fit.LensPositionFit(
                    positions=tracer.source_plane.positions,
                    noise_map=self.lens_data.pixel_scale,
                )

                if not fit.maximum_separation_within_threshold(
                    self.positions_threshold
                ):
                    raise exc.RayTracingException

        def check_inversion_pixels_are_below_limit(self, instance):

            if self.inversion_pixel_limit is not None:
                if instance.galaxies:
                    for galaxy in instance.galaxies:
                        if galaxy.has_pixelization:
                            if galaxy.pixelization.pixels > self.inversion_pixel_limit:
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

            tracer = self.tracer_for_instance(instance)

            phase_plotters.plot_ray_tracing_for_phase(
                tracer=tracer,
                during_analysis=during_analysis,
                mask=mask,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask,
                positions=positions,
                units=self.plot_units,
                should_plot_as_subplot=self.plot_ray_tracing_as_subplot,
                should_plot_all_at_end_png=self.plot_ray_tracing_all_at_end_png,
                should_plot_all_at_end_fits=self.plot_ray_tracing_all_at_end_fits,
                should_plot_image_plane_image=self.plot_ray_tracing_image_plane_image,
                should_plot_source_plane=self.plot_ray_tracing_source_plane,
                should_plot_convergence=self.plot_ray_tracing_convergence,
                should_plot_potential=self.plot_ray_tracing_potential,
                should_plot_deflections=self.plot_ray_tracing_deflections,
                visualize_path=image_path,
                subplot_path=subplot_path,
            )

            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_noise_background = self.hyper_noise_background_for_instance(
                instance=instance
            )

            fit = self.fit_for_tracer(
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_noise_background=hyper_noise_background,
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
