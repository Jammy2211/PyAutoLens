import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from autolens.array.util import binning_util
from autolens.lens import lens_data as ld
from autolens.lens import lens_fit
from autolens.model.galaxy import galaxy as g
from autolens.pipeline import phase_tagging
from autolens.pipeline.phase import phase_extensions
from autolens.pipeline.phase.phase_data import PhaseData
from autolens.plotters import visualizer


class PhaseImaging(PhaseData):
    galaxies = af.PhaseProperty("galaxies")
    hyper_image_sky = af.PhaseProperty("hyper_image_sky")
    hyper_background_noise = af.PhaseProperty("hyper_background_noise")

    def __init__(
        self,
        phase_name,
        phase_folders=tuple(),
        galaxies=None,
        hyper_image_sky=None,
        hyper_background_noise=None,
        optimizer_class=af.MultiNest,
        cosmology=cosmo.Planck15,
        sub_size=2,
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
        sub_size: int
            The side length of the subgrid
        pixel_scale_binned_cluster_grid : float or None
            If *True*, the hyper_galaxies image used to generate the cluster'grids weight map will be binned \
            up to this higher pixel scale to speed up the KMeans clustering algorithm.
        """

        phase_tag = phase_tagging.phase_tag_from_phase_settings(
            sub_size=sub_size,
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
            galaxies=galaxies,
            sub_size=sub_size,
            signal_to_noise_limit=signal_to_noise_limit,
            positions_threshold=positions_threshold,
            mask_function=mask_function,
            inner_mask_radii=inner_mask_radii,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            pixel_scale_binned_cluster_grid=pixel_scale_binned_cluster_grid,
            inversion_uses_border=inversion_uses_border,
            inversion_pixel_limit=inversion_pixel_limit,
            optimizer_class=optimizer_class,
            cosmology=cosmology,
            auto_link_priors=auto_link_priors,
        )

        self.bin_up_factor = bin_up_factor
        self.psf_shape = psf_shape

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.hyper_noise_map_max = af.conf.instance.general.get(
            "hyper", "hyper_noise_map_max", float
        )

        self.is_hyper_phase = False

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

    def make_analysis(self, data, results=None, mask=None, positions=None):
        """
        Create an lens object. Also calls the prior passing and lens_data modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        data: im.Imaging
            An lens_data that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """

        mask = self.setup_phase_mask(data=data, mask=mask)

        self.check_positions(positions=positions)

        pixel_scale_binned_grid = self.pixel_scale_binned_grid_from_mask(mask=mask)

        preload_pixelization_grids_of_planes = self.preload_pixelization_grids_of_planes_from_results(
            results=results
        )

        lens_imaging_data = ld.LensImagingData(
            imaging_data=data,
            mask=mask,
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
            image=lens_imaging_data.image(return_in_2d=True, return_masked=False),
            results=results,
        )

        lens_imaging_data = lens_imaging_data.new_lens_imaging_data_with_modified_image(
            modified_image=modified_image
        )

        if self.signal_to_noise_limit is not None:
            lens_imaging_data = lens_imaging_data.new_lens_imaging_data_with_signal_to_noise_limit(
                signal_to_noise_limit=self.signal_to_noise_limit
            )

        if self.bin_up_factor is not None:
            lens_imaging_data = lens_imaging_data.new_lens_imaging_data_with_binned_up_imaging_data_and_mask(
                bin_up_factor=self.bin_up_factor
            )

        self.output_phase_info()

        analysis = self.Analysis(
            lens_imaging_data=lens_imaging_data,
            cosmology=self.cosmology,
            image_path=self.optimizer.image_path,
            results=results,
        )

        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(self.optimizer.phase_output_path, "phase.info")

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.optimizer).__name__))
            phase_info.write("Sub-grid size = {} \n".format(self.sub_size))
            phase_info.write("PSF shape = {} \n".format(self.psf_shape))
            phase_info.write(
                "Positions Threshold = {} \n".format(self.positions_threshold)
            )
            phase_info.write("Cosmology = {} \n".format(self.cosmology))
            phase_info.write("Auto Link Priors = {} \n".format(self.auto_link_priors))

            phase_info.close()

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
    class Analysis(PhaseData.Analysis):
        def __init__(self, lens_imaging_data, cosmology, image_path=None, results=None):
            super(PhaseImaging.Analysis, self).__init__(
                cosmology=cosmology, results=results
            )
            self.visualizer = visualizer.PhaseImagingVisualizer(
                lens_imaging_data, image_path
            )

            self.lens_imaging_data = lens_imaging_data

            self.visualizer.plot_hyper_images(self.last_results)

            if self.last_results is not None:
                self.hyper_galaxy_image_1d_path_dict = (
                    self.last_results.hyper_galaxy_image_1d_path_dict
                )

                self.hyper_model_image_1d = self.last_results.hyper_model_image_1d

                self.binned_hyper_galaxy_image_1d_path_dict = self.last_results.binned_hyper_galaxy_image_1d_path_dict(
                    binned_grid=lens_imaging_data.grid.binned
                )

                self.visualizer.plot_hyper_images(self.last_results)

        @property
        def lens_data(self):
            return self.lens_imaging_data

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

            self.associate_images(instance=instance)
            tracer = self.tracer_for_instance(instance=instance)

            self.check_positions_trace_within_threshold_via_tracer(tracer=tracer)
            self.check_inversion_pixels_are_below_limit_via_tracer(tracer=tracer)

            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_background_noise = self.hyper_background_noise_for_instance(
                instance=instance
            )

            fit = self.lens_imaging_fit_for_tracer(
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            return fit.figure_of_merit

        def associate_images(self, instance: af.ModelInstance) -> af.ModelInstance:
            """
            Takes images from the last result, if there is one, and associates them with galaxies in this phase
            where full-path galaxy names match.

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

        def lens_imaging_fit_for_tracer(
            self, tracer, hyper_image_sky, hyper_background_noise
        ):

            return lens_fit.LensImagingFit.from_lens_data_and_tracer(
                lens_data=self.lens_imaging_data,
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

        def visualize(self, instance, during_analysis):
            instance = self.associate_images(instance=instance)
            tracer = self.tracer_for_instance(instance=instance)
            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)
            hyper_background_noise = self.hyper_background_noise_for_instance(
                instance=instance
            )

            fit = self.lens_imaging_fit_for_tracer(
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )
            self.visualizer.plot_ray_tracing(fit.tracer, during_analysis)
            self.visualizer.plot_lens_imaging(fit, during_analysis)

    class Result(PhaseData.Result):
        @property
        def most_likely_fit(self):

            hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
                instance=self.constant
            )

            hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
                instance=self.constant
            )

            return self.analysis.lens_imaging_fit_for_tracer(
                tracer=self.most_likely_tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

        @property
        def unmasked_model_image(self):
            return self.most_likely_fit.unmasked_blurred_profile_image

        @property
        def unmasked_model_image_of_planes(self):
            return self.most_likely_fit.unmasked_blurred_profile_image_of_planes

        @property
        def unmasked_model_image_of_planes_and_galaxies(self):
            fit = self.most_likely_fit
            return fit.unmasked_blurred_profile_image_of_planes_and_galaxies

        def image_2d_for_galaxy(self, galaxy: g.Galaxy) -> np.ndarray:
            """
            Parameters
            ----------
            galaxy
                A galaxy used in this phase

            Returns
            -------
            ndarray or None
                A numpy array giving the model image of that galaxy
            """
            return self.most_likely_fit.galaxy_model_image_2d_dict[galaxy]

        @property
        def image_galaxy_1d_dict(self) -> {str: g.Galaxy}:
            """
            A dictionary associating galaxy names with model images of those galaxies
            """

            image_1d_dict = {}

            for galaxy, galaxy_image_2d in self.image_galaxy_2d_dict.items():
                image_1d_dict[galaxy] = self.mask.mapping.array_1d_from_array_2d(
                    array_2d=galaxy_image_2d
                )

            return image_1d_dict

        @property
        def image_galaxy_2d_dict(self) -> {str: g.Galaxy}:
            """
            A dictionary associating galaxy names with model images of those galaxies
            """
            return {
                galaxy_path: self.image_2d_for_galaxy(galaxy)
                for galaxy_path, galaxy in self.path_galaxy_tuples
            }

        @property
        def hyper_galaxy_image_1d_path_dict(self):
            """
            A dictionary associating 1D hyper_galaxies galaxy images with their names.
            """

            hyper_minimum_percent = af.conf.instance.general.get(
                "hyper", "hyper_minimum_percent", float
            )

            hyper_galaxy_image_1d_path_dict = {}

            for path, galaxy in self.path_galaxy_tuples:

                galaxy_image_1d = self.image_galaxy_1d_dict[path]

                if not np.all(galaxy_image_1d == 0):
                    minimum_galaxy_value = hyper_minimum_percent * max(galaxy_image_1d)
                    galaxy_image_1d[
                        galaxy_image_1d < minimum_galaxy_value
                    ] = minimum_galaxy_value

                hyper_galaxy_image_1d_path_dict[path] = galaxy_image_1d

            return hyper_galaxy_image_1d_path_dict

        @property
        def hyper_galaxy_image_2d_path_dict(self):
            """
            A dictionary associating 2D hyper_galaxies galaxy images with their names.
            """

            hyper_galaxy_image_2d_path_dict = {}

            for path, galaxy in self.path_galaxy_tuples:
                hyper_galaxy_image_2d_path_dict[
                    path
                ] = self.mask.mapping.scaled_array_2d_from_array_1d(
                    array_1d=self.hyper_galaxy_image_1d_path_dict[path]
                )

            return hyper_galaxy_image_2d_path_dict

        def binned_image_1d_dict_from_binned_grid(self, binned_grid) -> {str: g.Galaxy}:
            """
            A dictionary associating 1D binned images with their names.
            """

            binned_image_1d_dict = {}

            for galaxy, galaxy_image_2d in self.image_galaxy_2d_dict.items():
                binned_image_2d = binning_util.binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(
                    array_2d=galaxy_image_2d, bin_up_factor=binned_grid.bin_up_factor
                )

                binned_image_1d_dict[
                    galaxy
                ] = binned_grid.mask.mapping.array_1d_from_array_2d(
                    array_2d=binned_image_2d
                )

            return binned_image_1d_dict

        def binned_hyper_galaxy_image_1d_path_dict(self, binned_grid):
            """
            A dictionary associating 1D hyper_galaxies galaxy binned images with their names.
            """

            if binned_grid is not None:

                hyper_minimum_percent = af.conf.instance.general.get(
                    "hyper", "hyper_minimum_percent", float
                )

                binned_image_1d_galaxy_dict = self.binned_image_1d_dict_from_binned_grid(
                    binned_grid=binned_grid
                )

                binned_hyper_galaxy_image_path_dict = {}

                for path, galaxy in self.path_galaxy_tuples:
                    binned_galaxy_image_1d = binned_image_1d_galaxy_dict[path]

                    minimum_hyper_value = hyper_minimum_percent * max(
                        binned_galaxy_image_1d
                    )
                    binned_galaxy_image_1d[
                        binned_galaxy_image_1d < minimum_hyper_value
                    ] = minimum_hyper_value

                    binned_hyper_galaxy_image_path_dict[path] = binned_galaxy_image_1d

                return binned_hyper_galaxy_image_path_dict

        def binned_hyper_galaxy_image_2d_path_dict(self, binned_grid):
            """
            A dictionary associating "D hyper_galaxies galaxy images binned images with their names.
            """

            if binned_grid is not None:

                binned_hyper_galaxy_image_1d_path_dict = self.binned_hyper_galaxy_image_1d_path_dict(
                    binned_grid=binned_grid
                )

                binned_hyper_galaxy_image_2d_path_dict = {}

                for path, galaxy in self.path_galaxy_tuples:
                    binned_hyper_galaxy_image_2d_path_dict[
                        path
                    ] = binned_grid.mask.mapping.scaled_array_2d_from_array_1d(
                        array_1d=binned_hyper_galaxy_image_1d_path_dict[path]
                    )

                return binned_hyper_galaxy_image_2d_path_dict

        @property
        def hyper_model_image_1d(self):

            hyper_model_image_1d = np.zeros(self.mask.pixels_in_mask)

            for path, galaxy in self.path_galaxy_tuples:
                hyper_model_image_1d += self.hyper_galaxy_image_1d_path_dict[path]

            return hyper_model_image_1d
