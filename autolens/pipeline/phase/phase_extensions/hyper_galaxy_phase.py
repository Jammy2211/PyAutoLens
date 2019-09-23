import copy

import numpy as np
from typing import cast

import autofit as af
from autolens.lens import lens_data as ld
from autolens.lens.lens_fit import lens_imaging_fit
from autolens.model.galaxy import galaxy as g
from autolens.model.hyper import hyper_data as hd
from autolens.pipeline.phase import phase_imaging
from autolens.plotters import visualizer
from .hyper_phase import HyperPhase


class HyperGalaxyPhase(HyperPhase):
    def __init__(self, phase):

        super().__init__(phase=phase, hyper_name="hyper_galaxy")
        self.include_sky_background = False
        self.include_noise_background = False

    class Analysis(af.Analysis):
        def __init__(
            self, lens_data, hyper_model_image_1d, hyper_galaxy_image_1d, image_path
        ):
            """
            An analysis to fit the noise for a single galaxy image.
            Parameters
            ----------
            lens_data: LensData
                lens data, including an image and noise
            hyper_model_image_1d: ndarray
                An image produce of the overall system by a model
            hyper_galaxy_image_1d: ndarray
                The contribution of one galaxy to the model image
            """

            self.lens_data = lens_data

            self.visualizer = visualizer.HyperGalaxyVisualizer(image_path)

            self.hyper_model_image_1d = hyper_model_image_1d
            self.hyper_galaxy_image_1d = hyper_galaxy_image_1d

        def visualize(self, instance, during_analysis):

            if self.visualizer.plot_hyper_galaxy_subplot:
                hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

                hyper_background_noise = self.hyper_background_noise_for_instance(
                    instance=instance
                )

                hyper_model_image_2d = self.lens_data.grid.mapping.scaled_array_2d_from_array_1d(
                    array_1d=self.hyper_model_image_1d
                )

                hyper_galaxy_image_2d = self.lens_data.grid.mapping.scaled_array_2d_from_array_1d(
                    array_1d=self.hyper_galaxy_image_1d
                )

                contribution_map_2d = instance.hyper_galaxy.contribution_map_from_hyper_images(
                    hyper_model_image=hyper_model_image_2d,
                    hyper_galaxy_image=hyper_galaxy_image_2d,
                )

                fit_normal = lens_imaging_fit.ImagingFit(
                    image=self.lens_data._image_1d,
                    noise_map=self.lens_data._noise_map_1d,
                    mask=self.lens_data._mask_1d,
                    model_image=self.hyper_model_image_1d,
                    mapping=self.lens_data.mapping,
                    inversion=None,
                )

                fit = self.fit_for_hyper_galaxy(
                    hyper_galaxy=instance.hyper_galaxy,
                    hyper_image_sky=hyper_image_sky,
                    hyper_background_noise=hyper_background_noise,
                )

                self.visualizer.hyper_galaxy_subplot(
                    hyper_galaxy_image=hyper_galaxy_image_2d,
                    contribution_map=contribution_map_2d,
                    noise_map=self.lens_data.noise_map(return_in_2d=True),
                    hyper_noise_map=fit.noise_map(return_in_2d=True),
                    chi_squared_map=fit_normal.chi_squared_map(return_in_2d=True),
                    hyper_chi_squared_map=fit.chi_squared_map(return_in_2d=True),
                )

        def fit(self, instance):
            """
            Fit the model image to the real image by scaling the hyper_galaxies noise.
            Parameters
            ----------
            instance: ModelInstance
                A model instance with a hyper_galaxies galaxy property
            Returns
            -------
            fit: float
            """

            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_background_noise = self.hyper_background_noise_for_instance(
                instance=instance
            )

            fit = self.fit_for_hyper_galaxy(
                hyper_galaxy=instance.hyper_galaxy,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            return fit.figure_of_merit

        @staticmethod
        def hyper_image_sky_for_instance(instance):
            if hasattr(instance, "hyper_image_sky"):
                return instance.hyper_image_sky

        @staticmethod
        def hyper_background_noise_for_instance(instance):
            if hasattr(instance, "hyper_background_noise"):
                return instance.hyper_background_noise

        def fit_for_hyper_galaxy(
            self, hyper_galaxy, hyper_image_sky, hyper_background_noise
        ):

            image_1d = lens_imaging_fit.image_1d_from_lens_data_and_hyper_image_sky(
                lens_data=self.lens_data, hyper_image_sky=hyper_image_sky
            )

            if hyper_background_noise is not None:
                noise_map_1d = hyper_background_noise.noise_map_scaled_noise_from_noise_map(
                    noise_map=self.lens_data._noise_map_1d
                )
            else:
                noise_map_1d = self.lens_data._noise_map_1d

            hyper_noise_map_1d = hyper_galaxy.hyper_noise_map_from_hyper_images_and_noise_map(
                hyper_model_image=self.hyper_model_image_1d,
                hyper_galaxy_image=self.hyper_galaxy_image_1d,
                noise_map=self.lens_data._noise_map_1d,
            )

            noise_map_1d = noise_map_1d + hyper_noise_map_1d
            if self.lens_data.hyper_noise_map_max is not None:
                noise_map_1d[
                    noise_map_1d > self.lens_data.hyper_noise_map_max
                ] = self.lens_data.hyper_noise_map_max

            return lens_imaging_fit.ImagingFit(
                image=image_1d,
                noise_map=noise_map_1d,
                mask=self.lens_data._mask_1d,
                model_image=self.hyper_model_image_1d,
                mapping=self.lens_data.mapping,
                inversion=None,
            )

        @classmethod
        def describe(cls, instance):
            return "Running hyper_galaxies galaxy fit for HyperGalaxy:\n{}".format(
                instance.hyper_galaxy
            )

    def run_hyper(self, data, results=None):
        """
        Run a fit for each galaxy from the previous phase.
        Parameters
        ----------
        data: LensData
        results: ResultsCollection
            Results from all previous phases
        Returns
        -------
        results: HyperGalaxyResults
            A collection of results, with one item per a galaxy
        """
        phase = self.make_hyper_phase()

        lens_data = ld.LensImagingData(
            imaging_data=data,
            mask=results.last.mask,
            trimmed_psf_shape=cast(phase_imaging.PhaseImaging, phase).psf_shape,
            positions=results.last.positions,
            positions_threshold=cast(
                phase_imaging.PhaseImaging, phase
            ).positions_threshold,
            pixel_scale_interpolation_grid=cast(
                phase_imaging.PhaseImaging, phase
            ).pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=cast(
                phase_imaging.PhaseImaging, phase
            ).pixel_scale_binned_cluster_grid,
            inversion_pixel_limit=cast(
                phase_imaging.PhaseImaging, phase
            ).inversion_pixel_limit,
            inversion_uses_border=cast(
                phase_imaging.PhaseImaging, phase
            ).inversion_uses_border,
            hyper_noise_map_max=cast(
                phase_imaging.PhaseImaging, phase
            ).hyper_noise_map_max,
            preload_pixelization_grids_of_planes=None,
        )

        model_image_1d = results.last.hyper_model_image_1d
        hyper_galaxy_image_1d_path_dict = results.last.hyper_galaxy_image_1d_path_dict

        hyper_result = copy.deepcopy(results.last)
        hyper_result.variable = hyper_result.variable.copy_with_fixed_priors(
            hyper_result.constant
        )

        hyper_result.analysis.hyper_model_image_1d = model_image_1d
        hyper_result.analysis.hyper_galaxy_image_1d_path_dict = (
            hyper_galaxy_image_1d_path_dict
        )

        for path, galaxy in results.last.path_galaxy_tuples:

            optimizer = phase.optimizer.copy_with_name_extension(extension=path[-1])

            optimizer.phase_tag = ""

            # TODO : This is a HACK :O

            optimizer.variable.galaxies = []

            optimizer.const_efficiency_mode = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_const_efficiency_mode", bool
            )
            optimizer.sampling_efficiency = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_sampling_efficiency", float
            )
            optimizer.n_live_points = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_n_live_points", int
            )
            optimizer.multimodal = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_multimodal", bool
            )

            optimizer.variable.hyper_galaxy = g.HyperGalaxy

            if self.include_sky_background:
                optimizer.variable.hyper_image_sky = hd.HyperImageSky

            if self.include_noise_background:
                optimizer.variable.hyper_background_noise = hd.HyperBackgroundNoise

            # If array is all zeros, galaxy did not have image in previous phase and
            # should be ignored
            if not np.all(hyper_galaxy_image_1d_path_dict[path] == 0):

                analysis = self.Analysis(
                    lens_data=lens_data,
                    hyper_model_image_1d=model_image_1d,
                    hyper_galaxy_image_1d=hyper_galaxy_image_1d_path_dict[path],
                    image_path=optimizer.image_path,
                )

                result = optimizer.fit(analysis)

                def transfer_field(name):
                    if hasattr(result.constant, name):
                        setattr(
                            hyper_result.constant.object_for_path(path),
                            name,
                            getattr(result.constant, name),
                        )
                        setattr(
                            hyper_result.variable.object_for_path(path),
                            name,
                            getattr(result.variable, name),
                        )

                transfer_field("hyper_galaxy")

                hyper_result.constant.hyper_image_sky = getattr(
                    result.constant, "hyper_image_sky"
                )
                hyper_result.variable.hyper_image_sky = getattr(
                    result.variable, "hyper_image_sky"
                )

                hyper_result.constant.hyper_background_noise = getattr(
                    result.constant, "hyper_background_noise"
                )
                hyper_result.variable.hyper_background_noise = getattr(
                    result.variable, "hyper_background_noise"
                )

        return hyper_result


class HyperGalaxyBackgroundSkyPhase(HyperGalaxyPhase):
    def __init__(self, phase):
        super().__init__(phase=phase)
        self.include_sky_background = True
        self.include_noise_background = False


class HyperGalaxyBackgroundNoisePhase(HyperGalaxyPhase):
    def __init__(self, phase):
        super().__init__(phase=phase)
        self.include_sky_background = False
        self.include_noise_background = True


class HyperGalaxyBackgroundBothPhase(HyperGalaxyPhase):
    def __init__(self, phase):
        super().__init__(phase=phase)
        self.include_sky_background = True
        self.include_noise_background = True


class HyperGalaxyAllPhase(HyperPhase):
    def __init__(
        self, phase, include_sky_background=False, include_noise_background=False
    ):
        super().__init__(phase=phase, hyper_name="hyper_galaxy")
        self.include_sky_background = include_sky_background
        self.include_noise_background = include_noise_background

    def run_hyper(self, data, results=None):
        """
        Run a fit for each galaxy from the previous phase.
        Parameters
        ----------
        data: LensData
        results: ResultsCollection
            Results from all previous phases
        Returns
        -------
        results: HyperGalaxyResults
            A collection of results, with one item per a galaxy
        """
        phase = self.make_hyper_phase()

        lens_data = ld.LensImagingData(
            imaging_data=data,
            mask=results.last.mask,
            trimmed_psf_shape=cast(phase_imaging.PhaseImaging, phase).psf_shape,
            positions=results.last.positions,
            positions_threshold=cast(
                phase_imaging.PhaseImaging, phase
            ).positions_threshold,
            pixel_scale_interpolation_grid=cast(
                phase_imaging.PhaseImaging, phase
            ).pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=cast(
                phase_imaging.PhaseImaging, phase
            ).pixel_scale_binned_cluster_grid,
            inversion_pixel_limit=cast(
                phase_imaging.PhaseImaging, phase
            ).inversion_pixel_limit,
            inversion_uses_border=cast(
                phase_imaging.PhaseImaging, phase
            ).inversion_uses_border,
            hyper_noise_map_max=cast(
                phase_imaging.PhaseImaging, phase
            ).hyper_noise_map_max,
            preload_pixelization_grids_of_planes=None,
        )

        model_image_1d = results.last.hyper_model_image_1d
        hyper_galaxy_image_1d_path_dict = results.last.hyper_galaxy_image_1d_path_dict

        hyper_result = copy.deepcopy(results.last)
        hyper_result.variable = hyper_result.variable.copy_with_fixed_priors(
            hyper_result.constant
        )

        hyper_result.analysis.hyper_model_image_1d = model_image_1d
        hyper_result.analysis.hyper_galaxy_image_1d_path_dict = (
            hyper_galaxy_image_1d_path_dict
        )

        for path, galaxy in results.last.path_galaxy_tuples:

            optimizer = phase.optimizer.copy_with_name_extension(extension=path[-1])

            optimizer.phase_tag = ""

            # TODO : This is a HACK :O

            optimizer.variable.galaxies = []
            optimizer.variable.galaxies = []
            optimizer.variable.galaxies = []

            optimizer.const_efficiency_mode = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_const_efficiency_mode", bool
            )
            optimizer.sampling_efficiency = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_sampling_efficiency", float
            )
            optimizer.n_live_points = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_n_live_points", int
            )

            optimizer.variable.hyper_galaxy = g.HyperGalaxy

            if self.include_sky_background:
                optimizer.variable.hyper_image_sky = hd.HyperImageSky

            if self.include_noise_background:
                optimizer.variable.hyper_background_noise = hd.HyperBackgroundNoise

            # If array is all zeros, galaxy did not have image in previous phase and
            # should be ignored
            if not np.all(hyper_galaxy_image_1d_path_dict[path] == 0):

                analysis = self.Analysis(
                    lens_data=lens_data,
                    model_image_1d=model_image_1d,
                    galaxy_image_1d=hyper_galaxy_image_1d_path_dict[path],
                )

                result = optimizer.fit(analysis)

                def transfer_field(name):
                    if hasattr(result.constant, name):
                        setattr(
                            hyper_result.constant.object_for_path(path),
                            name,
                            getattr(result.constant, name),
                        )
                        setattr(
                            hyper_result.variable.object_for_path(path),
                            name,
                            getattr(result.variable, name),
                        )

                transfer_field("hyper_galaxy")
                transfer_field("hyper_image_sky")
                transfer_field("hyper_background_noise")

        return hyper_result
