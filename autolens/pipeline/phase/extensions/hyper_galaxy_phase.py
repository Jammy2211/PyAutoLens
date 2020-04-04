import copy
import pickle
from typing import cast

import numpy as np

import autofit as af
from autoarray.fit import fit as aa_fit
from autoastro.galaxy import galaxy as g
from autoastro.hyper import hyper_data as hd
from autolens.dataset import imaging
from autolens.fit import fit
from autolens.pipeline import visualizer
from autolens.pipeline.phase.imaging import PhaseImaging
from .hyper_phase import HyperPhase


class Analysis(af.Analysis):
    def __init__(
        self, masked_imaging, hyper_model_image, hyper_galaxy_image, image_path
    ):
        """
        An analysis to fit the noise for a single galaxy image.
        Parameters
        ----------
        masked_imaging: LensData
            lens dataset, including an image and noise
        hyper_model_image: ndarray
            An image produce of the overall system by a model
        hyper_galaxy_image: ndarray
            The contribution of one galaxy to the model image
        """

        self.masked_imaging = masked_imaging

        self.visualizer = visualizer.HyperGalaxyVisualizer(image_path=image_path)

        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_image = hyper_galaxy_image

    def visualize(self, instance, during_analysis):

        if self.visualizer.plot_hyper_galaxy_subplot:
            hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

            hyper_background_noise = self.hyper_background_noise_for_instance(
                instance=instance
            )

            contribution_map = instance.hyper_galaxy.contribution_map_from_hyper_images(
                hyper_model_image=self.hyper_model_image,
                hyper_galaxy_image=self.hyper_galaxy_image,
            )

            fit_normal = aa_fit.FitImaging(
                masked_imaging=self.masked_imaging, model_image=self.hyper_model_image
            )

            fit_hyper = self.fit_for_hyper_galaxy(
                hyper_galaxy=instance.hyper_galaxy,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            self.visualizer.visualize_hyper_galaxy(
                fit=fit_normal,
                hyper_fit=fit_hyper,
                galaxy_image=self.hyper_galaxy_image,
                contribution_map_in=contribution_map,
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

        image = fit.hyper_image_from_image_and_hyper_image_sky(
            image=self.masked_imaging.image, hyper_image_sky=hyper_image_sky
        )

        if hyper_background_noise is not None:
            noise_map = hyper_background_noise.hyper_noise_map_from_noise_map(
                noise_map=self.masked_imaging.noise_map
            )
        else:
            noise_map = self.masked_imaging.noise_map

        hyper_noise_map = hyper_galaxy.hyper_noise_map_from_hyper_images_and_noise_map(
            hyper_model_image=self.hyper_model_image,
            hyper_galaxy_image=self.hyper_galaxy_image,
            noise_map=self.masked_imaging.noise_map,
        )

        noise_map = noise_map + hyper_noise_map

        masked_imaging = self.masked_imaging.modify_image_and_noise_map(
            image=image, noise_map=noise_map
        )

        return aa_fit.FitImaging(
            masked_imaging=masked_imaging, model_image=self.hyper_model_image
        )

    @classmethod
    def describe(cls, instance):
        return "Running hyper_galaxies galaxy fit for HyperGalaxy:\n{}".format(
            instance.hyper_galaxy
        )


class HyperGalaxyPhase(HyperPhase):
    Analysis = Analysis

    def __init__(self, phase):

        super().__init__(phase=phase, hyper_name="hyper_galaxy")
        self.include_sky_background = False
        self.include_noise_background = False

    def run_hyper(self, dataset, results=None):
        """
        Run a fit for each galaxy from the previous phase.
        Parameters
        ----------
        dataset: LensData
        results: ResultsCollection
            Results from all previous phases
        Returns
        -------
        results: HyperGalaxyResults
            A collection of results, with one item per a galaxy
        """

        phase = self.make_hyper_phase()

        masked_imaging = imaging.MaskedImaging(
            imaging=dataset,
            mask=results.last.mask,
            psf_shape_2d=dataset.psf.shape_2d,
            positions=results.last.positions,
            positions_threshold=cast(
                PhaseImaging, phase
            ).meta_dataset.positions_threshold,
            pixel_scale_interpolation_grid=cast(
                PhaseImaging, phase
            ).meta_dataset.pixel_scale_interpolation_grid,
            inversion_pixel_limit=cast(
                PhaseImaging, phase
            ).meta_dataset.inversion_pixel_limit,
            inversion_uses_border=cast(
                PhaseImaging, phase
            ).meta_dataset.inversion_uses_border,
            preload_sparse_grids_of_planes=None,
        )

        hyper_result = copy.deepcopy(results.last)
        hyper_result.model = hyper_result.model.copy_with_fixed_priors(
            hyper_result.instance
        )

        hyper_result.analysis.hyper_model_image = results.last.hyper_model_image
        hyper_result.analysis.hyper_galaxy_image_path_dict = (
            results.last.hyper_galaxy_image_path_dict
        )

        for path, galaxy in results.last.path_galaxy_tuples:

            # TODO : NEed t be sure these wont mess up anything else.

            optimizer = phase.optimizer.copy_with_name_extension(extension=path[-1])

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
            optimizer.evidence_tolerance = af.conf.instance.non_linear.get(
                "MultiNest", "extension_hyper_galaxy_evidence_tolerance", float
            )

            model = copy.deepcopy(phase.model)

            # TODO : This is a HACK :O

            model.galaxies = []

            model.hyper_galaxy = g.HyperGalaxy

            if self.include_sky_background:
                model.hyper_image_sky = hd.HyperImageSky

            if self.include_noise_background:
                model.hyper_background_noise = hd.HyperBackgroundNoise

            # If arrays is all zeros, galaxy did not have image in previous phase and
            # shoumasked_imaging be ignored
            if not np.all(
                hyper_result.analysis.hyper_galaxy_image_path_dict[path] == 0
            ):
                hyper_model_image = hyper_result.analysis.hyper_model_image

                analysis = self.Analysis(
                    masked_imaging=masked_imaging,
                    hyper_model_image=hyper_model_image,
                    hyper_galaxy_image=hyper_result.analysis.hyper_galaxy_image_path_dict[
                        path
                    ],
                    image_path=optimizer.paths.image_path,
                )

                result = optimizer.fit(analysis=analysis, model=model)

                def transfer_field(name):
                    if hasattr(result.instance, name):
                        setattr(
                            hyper_result.instance.object_for_path(path),
                            name,
                            getattr(result.instance, name),
                        )
                        setattr(
                            hyper_result.model.object_for_path(path),
                            name,
                            getattr(result.model, name),
                        )

                transfer_field("hyper_galaxy")

                hyper_result.instance.hyper_image_sky = getattr(
                    result.instance, "hyper_image_sky"
                )
                hyper_result.model.hyper_image_sky = getattr(
                    result.model, "hyper_image_sky"
                )

                hyper_result.instance.hyper_background_noise = getattr(
                    result.instance, "hyper_background_noise"
                )
                hyper_result.model.hyper_background_noise = getattr(
                    result.model, "hyper_background_noise"
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
