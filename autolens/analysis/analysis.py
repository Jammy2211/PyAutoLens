from astropy import cosmology as cosmo
import copy
import json
import numpy as np
from os import path
from scipy.stats import norm
from typing import List

from autoconf import conf
import autofit as af
import autoarray as aa
import autogalaxy as ag

from autogalaxy.analysis.analysis import AnalysisDataset as AgAnalysisDataset

from autolens import exc
from autolens.analysis.result import ResultInterferometer
from autolens.analysis.result import ResultPoint
from autolens.analysis.visualizer import Visualizer
from autolens.dataset.point_dataset import PointDict

from autolens.fit.fit_interferometer import FitInterferometer
from autolens.fit.fit_point import FitPointDict
from autolens.lens.positions_solver import PositionsSolver
from autolens.lens.ray_tracing import Tracer
from autolens.lens.settings import SettingsLens


class AnalysisLensing:
    def __init__(self, settings_lens=SettingsLens(), cosmology=cosmo.Planck15):

        self.cosmology = cosmology
        self.settings_lens = settings_lens

    def tracer_for_instance(self, instance):

        if hasattr(instance, "perturbation"):
            instance.galaxies.subhalo = instance.perturbation

        return Tracer.from_galaxies(
            galaxies=instance.galaxies, cosmology=self.cosmology
        )


class AnalysisDataset(AgAnalysisDataset, AnalysisLensing):
    def __init__(
        self,
        dataset,
        positions: aa.Grid2DIrregular = None,
        hyper_dataset_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        settings_lens=SettingsLens(),
        preloads=aa.Preloads(),
    ):
        """

        Parameters
        ----------
        dataset
        positions : aa.Grid2DIrregular
            Image-pixel coordinates in arc-seconds of bright regions of the lensed source that will map close to one
            another in the source-plane(s) for an accurate mass model, which can be used to discard unphysical mass
            models during model-fitting.
        cosmology
        settings_pixelization
        settings_inversion
        settings_lens
        preloads
        """

        super().__init__(
            dataset=dataset,
            hyper_dataset_result=hyper_dataset_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
        )

        AnalysisLensing.__init__(
            self=self, settings_lens=settings_lens, cosmology=cosmology
        )

        self.positions = positions

        self.settings_lens = settings_lens

    def log_likelihood_cap_from(self, stochastic_log_evidences_json_file):

        try:
            with open(stochastic_log_evidences_json_file, "r") as f:
                stochastic_log_evidences = np.asarray(json.load(f))
        except FileNotFoundError:
            raise exc.AnalysisException(
                "The file 'stochastic_log_evidences.json' could not be found in the output of the model-fitting results"
                "in the analysis before the stochastic analysis. Rerun PyAutoLens with `stochastic_outputs=True` in the"
                "`general.ini` configuration file."
            )

        mean, sigma = norm.fit(stochastic_log_evidences)

        return mean

    def stochastic_log_evidences_for_instance(self, instance) -> List[float]:
        raise NotImplementedError()

    def save_settings(self, paths: af.DirectoryPaths):

        super().save_settings(paths=paths)

        paths.save_object("settings_lens", self.settings_lens)

    def save_stochastic_outputs(
        self, paths: af.DirectoryPaths, samples: af.OptimizerSamples
    ):

        stochastic_log_evidences_json_file = path.join(
            paths.output_path, "stochastic_log_evidences.json"
        )

        try:
            with open(stochastic_log_evidences_json_file, "r") as f:
                stochastic_log_evidences = np.asarray(json.load(f))
        except FileNotFoundError:
            instance = samples.max_log_likelihood_instance
            stochastic_log_evidences = self.stochastic_log_evidences_for_instance(
                instance=instance
            )

        if stochastic_log_evidences is None:
            return

        with open(stochastic_log_evidences_json_file, "w") as outfile:
            json.dump(
                [float(evidence) for evidence in stochastic_log_evidences], outfile
            )

        paths.save_object("stochastic_log_evidences", stochastic_log_evidences)

        visualizer = Visualizer(visualize_path=paths.image_path)

        visualizer.visualize_stochastic_histogram(
            log_evidences=stochastic_log_evidences,
            max_log_evidence=np.max(samples.log_likelihood_list),
            histogram_bins=self.settings_lens.stochastic_histogram_bins,
        )

    @property
    def no_positions(self):

        # settings_lens = SettingsLens(
        # positions_threshold=None,
        # stochastic_likelihood_resamples=self.settings_lens.stochastic_likelihood_resamples,
        # stochastic_samples = self.settings_lens.stochastic_samples,
        # stochastic_histogram_bins = self.settings_lens.stochastic_histogram_bins
        # )
        #
        # return self.__class__(
        #     dataset=self.dataset,
        #     positions = None,
        #     hyper_dataset_result=self.hyper_result,
        #     cosmology=self.cosmology,
        #     settings_pixelization=self.settings_pixelization,
        #     settings_inversion=self.settings_inversion,
        #     settings_lens=settings_lens,
        #     preloads=self.preloads
        # )

        analysis = copy.deepcopy(self)

        analysis.positions = None
        analysis.settings_lens.positions_threshold = None

        return analysis


class AnalysisInterferometer(AnalysisDataset):
    def __init__(
        self,
        dataset,
        positions: aa.Grid2DIrregular = None,
        hyper_dataset_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        settings_lens=SettingsLens(),
        preloads=aa.Preloads(),
    ):

        super().__init__(
            dataset=dataset,
            positions=positions,
            hyper_dataset_result=hyper_dataset_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            settings_lens=settings_lens,
            preloads=preloads,
        )

        if self.hyper_dataset_result is not None:

            self.set_hyper_dataset(result=self.hyper_dataset_result)

        else:

            self.hyper_galaxy_visibilities_path_dict = None
            self.hyper_model_visibilities = None

    @property
    def interferometer(self):
        return self.dataset

    def set_hyper_dataset(self, result):

        super().set_hyper_dataset(result=result)

        self.hyper_model_visibilities = result.hyper_model_visibilities
        self.hyper_galaxy_visibilities_path_dict = (
            result.hyper_galaxy_visibilities_path_dict
        )

    def log_likelihood_function(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the interferometer in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model interferometer itself
        """

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        self.settings_lens.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer, positions=self.positions
        )

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        try:
            fit = self.fit_interferometer_for_tracer(
                tracer=tracer, hyper_background_noise=hyper_background_noise
            )
            return fit.figure_of_merit
        except (
            exc.PixelizationException,
            exc.InversionException,
            exc.GridException,
            OverflowError,
        ) as e:
            raise exc.FitException from e

    def associate_hyper_visibilities(
        self, instance: af.ModelInstance
    ) -> af.ModelInstance:
        """
        Takes visibilities from the last result, if there is one, and associates them with galaxies in this search
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
           The input instance with visibilities associated with galaxies where possible.
        """
        if self.hyper_galaxy_visibilities_path_dict is not None:
            for galaxy_path, galaxy in instance.path_instance_tuples_for_class(
                ag.Galaxy
            ):
                if galaxy_path in self.hyper_galaxy_visibilities_path_dict:
                    galaxy.hyper_model_visibilities = self.hyper_model_visibilities
                    galaxy.hyper_galaxy_visibilities = self.hyper_galaxy_visibilities_path_dict[
                        galaxy_path
                    ]

        return instance

    def fit_interferometer_for_tracer(
        self, tracer, hyper_background_noise, use_hyper_scalings=True
    ):

        return FitInterferometer(
            interferometer=self.dataset,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scaling=use_hyper_scalings,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            preloads=self.preloads,
        )

    def stochastic_log_evidences_for_instance(self, instance):

        instance = self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        if not tracer.has_pixelization:
            return None

        if not isinstance(
            tracer.pixelizations_of_planes[-1], aa.pix.VoronoiBrightnessImage
        ):
            return None

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        settings_pixelization = (
            self.settings_pixelization.settings_with_is_stochastic_true()
        )

        log_evidences = []

        for i in range(self.settings_lens.stochastic_samples):

            try:
                log_evidence = FitInterferometer(
                    interferometer=self.dataset,
                    tracer=tracer,
                    hyper_background_noise=hyper_background_noise,
                    settings_pixelization=settings_pixelization,
                    settings_inversion=self.settings_inversion,
                    preloads=self.preloads,
                ).log_evidence
            except (
                exc.PixelizationException,
                exc.InversionException,
                exc.GridException,
                OverflowError,
            ) as e:
                log_evidence = None

            if log_evidence is not None:
                log_evidences.append(log_evidence)

        return log_evidences

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis):

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.fit_interferometer_for_tracer(
            tracer=tracer, hyper_background_noise=hyper_background_noise
        )

        visualizer = Visualizer(visualize_path=paths.image_path)
        visualizer.visualize_interferometer(interferometer=self.interferometer)
        visualizer.visualize_fit_interferometer(
            fit=fit, during_analysis=during_analysis
        )
        visualizer.visualize_tracer(
            tracer=fit.tracer, grid=fit.grid, during_analysis=during_analysis
        )
        if fit.inversion is not None:
            visualizer.visualize_inversion(
                inversion=fit.inversion, during_analysis=during_analysis
            )

        visualizer.visualize_hyper_images(
            hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
            hyper_model_image=self.hyper_model_image,
            tracer=tracer,
        )

        if visualizer.plot_fit_no_hyper:

            fit = self.fit_interferometer_for_tracer(
                tracer=tracer, hyper_background_noise=None, use_hyper_scalings=False
            )

            visualizer.visualize_fit_interferometer(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def save_results_for_aggregator(
        self,
        paths: af.DirectoryPaths,
        samples: af.OptimizerSamples,
        model: af.Collection,
    ):

        if conf.instance["general"]["hyper"]["stochastic_outputs"]:
            self.save_stochastic_outputs(paths=paths, samples=samples)

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return ResultInterferometer(
            samples=samples, model=model, analysis=self, search=search
        )

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        super().save_attributes_for_aggregator(paths=paths)

        paths.save_object("uv_wavelengths", self.dataset.uv_wavelengths)
        paths.save_object("real_space_mask", self.dataset.real_space_mask)
        paths.save_object("positions", self.positions)


class AnalysisPoint(af.Analysis, AnalysisLensing):
    def __init__(
        self,
        point_dict: PointDict,
        solver: PositionsSolver,
        imaging=None,
        cosmology=cosmo.Planck15,
        settings_lens=SettingsLens(),
    ):
        """
        The analysis performed for model-fitting a point-source dataset, for example fitting the point-sources of a
        multiply imaged lensed quasar or supernovae of many source galaxies of a galaxy cluster.

        The analysis brings together the data, model and non-linear search in the classes `log_likelihood_function`,
        which is called by every iteration of the non-linear search to compute a likelihood value which samples
        parameter space.

        Parameters
        ----------
        point_dict
            A dictionary containing the full point source dictionary that is used for model-fitting.
        solver
            The object which is used to determine the image-plane of source-plane positions of a model (via a `Tracer`).
        imaging
            The imaging of the point-source dataset, which is not used for model-fitting but can be used for
            visualization.
        cosmology
            The cosmology of the ray-tracing calculation.
        settings_lens
            Settings which control how the model-fit is performed.
        """

        super().__init__(settings_lens=settings_lens, cosmology=cosmology)

        AnalysisLensing.__init__(
            self=self, settings_lens=settings_lens, cosmology=cosmology
        )

        self.point_dict = point_dict

        self.solver = solver
        self.imaging = imaging

    def log_likelihood_function(self, instance):
        """
        Determine the fit of the strong lens system of lens galaxies and source galaxies to the point source data.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model masked_imaging itself
        """

        tracer = self.tracer_for_instance(instance=instance)

        fit = FitPointDict(
            point_dict=self.point_dict, tracer=tracer, positions_solver=self.solver
        )

        return fit.log_likelihood

    def visualize(self, paths, instance, during_analysis):

        tracer = self.tracer_for_instance(instance=instance)

        visualizer = Visualizer(visualize_path=paths.image_path)

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return ResultPoint(samples=samples, model=model, analysis=self, search=search)

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        paths.save_object("dataset", self.point_dict)
