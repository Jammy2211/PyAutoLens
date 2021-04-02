import numpy as np
from os import path
import pickle
from typing import List
import json
from scipy.stats import norm
from astropy import cosmology as cosmo
import numba

from autoconf import conf
import autofit as af
from autofit.exc import FitException
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.inversion import pixelizations as pix, inversions as inv
from autoarray import preloads as pload
from autoarray.exc import PixelizationException, InversionException, GridException
from autogalaxy.analysis import model_util
from autogalaxy.galaxy import galaxy as g
from autogalaxy.analysis import analysis as a
from autolens.lens import ray_tracing
from autolens.lens import positions_solver as psolve
from autolens.dataset import point_source as ps
from autolens.fit import fit_imaging
from autolens.fit import fit_interferometer
from autolens.fit import fit_point_source
from autolens.lens import settings
from autolens.analysis import result as res
from autolens.analysis import visualizer as vis
from autolens import exc


class AnalysisLensing:
    def __init__(self, settings_lens=settings.SettingsLens(), cosmology=cosmo.Planck15):

        self.cosmology = cosmology
        self.settings_lens = settings_lens

    def tracer_for_instance(self, instance):

        if hasattr(instance, "perturbation"):
            instance.galaxies.subhalo = instance.perturbation

        return ray_tracing.Tracer.from_galaxies(
            galaxies=instance.galaxies, cosmology=self.cosmology
        )


class AnalysisDataset(a.AnalysisDataset, AnalysisLensing):
    def __init__(
        self,
        dataset,
        positions: grid_2d_irregular.Grid2DIrregular = None,
        hyper_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        settings_lens=settings.SettingsLens(),
        preloads=pload.Preloads(),
        use_result_as_hyper_dataset=False,
    ):
        """

        Parameters
        ----------
        dataset
        positions : grid_2d_irregular.Grid2DIrregular
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
            hyper_result=hyper_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
            use_result_as_hyper_dataset=use_result_as_hyper_dataset,
        )

        AnalysisLensing.__init__(
            self=self, settings_lens=settings_lens, cosmology=cosmology
        )

        self.positions = positions

        self.settings_lens = settings_lens

    def modify_before_fit(self, model, paths: af.Paths):

        #    self.preloads = self.setup_preloads(model=model)

        return self

    def setup_preloads(self, model):

        return pload.Preloads()

        if self.results is None:
            return pload.Preloads()

        # Preload the source-plane grid of coordinates if the source parameters are fixed, skipping the KMeans.

        sparse_grids_of_planes = None
        blurred_mapping_matrix = None
        curvature_matrix = None
        mapper = None

        if (
            self.results.last is not None
            and model_util.pixelization_from(model=model) is not None
            and not model_util.pixelization_is_model_from(model=model)
        ):
            if model.pixelization.__class__ is self.results.last.pixelization.__class__:
                if hasattr(self.results.last, "hyper"):
                    sparse_grids_of_planes = (
                        self.results.last.hyper.max_log_likelihood_pixelization_grids_of_planes
                    )
                else:
                    sparse_grids_of_planes = (
                        self.results.last.max_log_likelihood_pixelization_grids_of_planes
                    )

        # Preload the blurred_mapping_matrix and curvature matrix calculation if only the parametric light profiles
        # are being fitted.

        # if self.preload_inversion:
        #
        #     if hasattr(results.last, "hyper"):
        #         inversion = results.last.hyper.max_log_likelihood_fit.inversion
        #     else:
        #         inversion = results.last.max_log_likelihood_fit.inversion
        #
        #     if inversion is not None:
        #
        #         blurred_mapping_matrix = inversion.blurred_mapping_matrix
        #         curvature_matrix = inversion.curvature_matrix
        #         mapper = inversion.mapper

        return pload.Preloads(
            sparse_grids_of_planes=sparse_grids_of_planes,
            mapper=mapper,
            blurred_mapping_matrix=blurred_mapping_matrix,
            curvature_matrix=curvature_matrix,
        )

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

    def save_settings(self, paths: af.Paths):

        super().save_settings(paths=paths)

        with open(path.join(paths.pickle_path, "settings_lens.pickle"), "wb+") as f:
            pickle.dump(self.settings_lens, f)

    def save_stochastic_outputs(self, paths: af.Paths, samples: af.OptimizerSamples):

        stochastic_log_evidences_json_file = path.join(
            paths.output_path, "stochastic_log_evidences.json"
        )
        stochastic_log_evidences_pickle_file = path.join(
            paths.pickle_path, "stochastic_log_evidences.pickle"
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

        with open(stochastic_log_evidences_pickle_file, "wb") as f:
            pickle.dump(stochastic_log_evidences, f)

        visualizer = vis.Visualizer(visualize_path=paths.image_path)

        visualizer.visualize_stochastic_histogram(
            log_evidences=stochastic_log_evidences,
            max_log_evidence=np.max(samples.log_likelihoods),
            histogram_bins=self.settings_lens.stochastic_histogram_bins,
        )


class AnalysisImaging(AnalysisDataset):
    @property
    def imaging(self):
        return self.dataset

    def log_likelihood_function(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the imaging in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model imaging itself
        """

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        self.settings_lens.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer, positions=self.positions
        )

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        try:
            return self.imaging_fit_for_tracer(
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            ).figure_of_merit
        except (
            PixelizationException,
            InversionException,
            GridException,
            OverflowError,
        ) as e:
            raise FitException from e

    def imaging_fit_for_tracer(
        self, tracer, hyper_image_sky, hyper_background_noise, use_hyper_scalings=True
    ):

        return fit_imaging.FitImaging(
            imaging=self.dataset,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
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
            return

        if not isinstance(
            tracer.pixelizations_of_planes[-1], pix.VoronoiBrightnessImage
        ):
            return

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        settings_pixelization = (
            self.settings_pixelization.settings_with_is_stochastic_true()
        )

        log_evidences = []

        for i in range(self.settings_lens.stochastic_samples):

            try:
                log_evidence = fit_imaging.FitImaging(
                    imaging=self.dataset,
                    tracer=tracer,
                    hyper_image_sky=hyper_image_sky,
                    hyper_background_noise=hyper_background_noise,
                    settings_pixelization=settings_pixelization,
                    settings_inversion=self.settings_inversion,
                    preloads=self.preloads,
                ).log_evidence
            except (
                PixelizationException,
                InversionException,
                GridException,
                OverflowError,
            ) as e:
                log_evidence = None

            if log_evidence is not None:
                log_evidences.append(log_evidence)

        return log_evidences

    def visualize(self, paths: af.Paths, instance, during_analysis):

        instance = self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)
        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.imaging_fit_for_tracer(
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        visualizer = vis.Visualizer(visualize_path=paths.image_path)

        visualizer.visualize_imaging(imaging=self.imaging)
        visualizer.visualize_fit_imaging(fit=fit, during_analysis=during_analysis)
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
            fit = self.imaging_fit_for_tracer(
                tracer=tracer,
                hyper_image_sky=None,
                hyper_background_noise=None,
                use_hyper_scalings=False,
            )

            visualizer.visualize_fit_imaging(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def save_results_for_aggregator(
        self, paths: af.Paths, samples: af.OptimizerSamples, model: af.Collection
    ):

        pixelization = model_util.pixelization_from(model=model)

        if conf.instance["general"]["hyper"]["stochastic_outputs"]:
            if model_util.isinstance_or_prior(pixelization, pix.VoronoiBrightnessImage):
                self.save_stochastic_outputs(paths=paths, samples=samples)

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return res.ResultImaging(
            samples=samples, model=model, analysis=self, search=search
        )

    def make_attributes(self):
        return AttributesImaging(
            cosmology=self.cosmology,
            positions=self.positions,
            hyper_model_image=self.hyper_model_image,
            hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
        )


class AnalysisInterferometer(AnalysisDataset):
    def __init__(
        self,
        dataset,
        positions: grid_2d_irregular.Grid2DIrregular = None,
        hyper_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        settings_lens=settings.SettingsLens(),
        preloads=pload.Preloads(),
        use_result_as_hyper_dataset=False,
    ):

        super().__init__(
            dataset=dataset,
            positions=positions,
            hyper_result=hyper_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            settings_lens=settings_lens,
            preloads=preloads,
            use_result_as_hyper_dataset=use_result_as_hyper_dataset,
        )

        if self.hyper_result is not None:

            self.set_hyper_dataset(result=self.hyper_result)

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
            fit = self.interferometer_fit_for_tracer(
                tracer=tracer, hyper_background_noise=hyper_background_noise
            )
            return fit.figure_of_merit
        except (
            PixelizationException,
            InversionException,
            GridException,
            OverflowError,
        ) as e:
            raise FitException from e

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
                g.Galaxy
            ):
                if galaxy_path in self.hyper_galaxy_visibilities_path_dict:
                    galaxy.hyper_model_visibilities = self.hyper_model_visibilities
                    galaxy.hyper_galaxy_visibilities = self.hyper_galaxy_visibilities_path_dict[
                        galaxy_path
                    ]

        return instance

    def interferometer_fit_for_tracer(
        self, tracer, hyper_background_noise, use_hyper_scalings=True
    ):

        return fit_interferometer.FitInterferometer(
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
            tracer.pixelizations_of_planes[-1], pix.VoronoiBrightnessImage
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
                log_evidence = fit_interferometer.FitInterferometer(
                    interferometer=self.dataset,
                    tracer=tracer,
                    hyper_background_noise=hyper_background_noise,
                    settings_pixelization=settings_pixelization,
                    settings_inversion=self.settings_inversion,
                    preloads=self.preloads,
                ).log_evidence
            except (
                PixelizationException,
                InversionException,
                GridException,
                OverflowError,
            ) as e:
                log_evidence = None

            if log_evidence is not None:
                log_evidences.append(log_evidence)

        return log_evidences

    def visualize(self, paths: af.Paths, instance, during_analysis):

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.interferometer_fit_for_tracer(
            tracer=tracer, hyper_background_noise=hyper_background_noise
        )

        visualizer = vis.Visualizer(visualize_path=paths.image_path)
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

            fit = self.interferometer_fit_for_tracer(
                tracer=tracer, hyper_background_noise=None, use_hyper_scalings=False
            )

            visualizer.visualize_fit_interferometer(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def save_results_for_aggregator(
        self, paths: af.Paths, samples: af.OptimizerSamples, model: af.Collection
    ):

        if conf.instance["general"]["hyper"]["stochastic_outputs"]:
            self.save_stochastic_outputs(paths=paths, samples=samples)

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return res.ResultInterferometer(
            samples=samples, model=model, analysis=self, search=search
        )

    def make_attributes(self):
        return AttributesInterferometer(
            cosmology=self.cosmology,
            positions=self.positions,
            real_space_mask=self.dataset.real_space_mask,
            hyper_model_image=self.hyper_model_image,
            hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
        )


class AnalysisPointSource(af.Analysis, AnalysisLensing):
    def __init__(
        self,
        point_source_dict: ps.PointSourceDict,
        solver: psolve.PositionsSolver,
        imaging=None,
        cosmology=cosmo.Planck15,
        settings_lens=settings.SettingsLens(),
    ):
        """
        The analysis performed for model-fitting a point-source dataset, for example fitting the point-sources of a
        multiply imaged lensed quasar or supernovae of many source galaxies of a galaxy cluster.

        The analysis brings together the data, model and non-linear search in the classes `log_likelihood_function`,
        which is called by every iteration of the non-linear search to compute a likelihood value which samples
        parameter space.

        Parameters
        ----------
        point_source_dict : ps.PointSourceDict
            A dictionary containing the full point source dictionary that is used for model-fitting.
        solver : psolve.PositionsSolver
            The object which is used to determine the image-plane of source-plane positions of a model (via a `Tracer`).
        imaging : Imaging
            The imaging of the point-source dataset, which is not used for model-fitting but can be used for
            visualization.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        settings_lens : settings.SettingsLens()
            Settings which control how the model-fit is performed.
        """

        super().__init__(settings_lens=settings_lens, cosmology=cosmology)

        AnalysisLensing.__init__(
            self=self, settings_lens=settings_lens, cosmology=cosmology
        )

        self.point_source_dict = point_source_dict

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

        log_likelihood = 0.0

        for point_source_dataset in self.point_source_dict.values():

            try:
                fit_positions = self.fit_positions_for(
                    point_source_dataset=point_source_dataset, tracer=tracer
                )
            except (AttributeError, numba.errors.TypingError) as e:
                raise FitException from e

            log_likelihood += fit_positions.log_likelihood

            if point_source_dataset.fluxes is not None:
                fit_fluxes = self.fit_fluxes_for(
                    point_source_dataset=point_source_dataset, tracer=tracer
                )
                log_likelihood += fit_fluxes.log_likelihood

        return log_likelihood

    def fit_positions_for(self, point_source_dataset, tracer):

        return fit_point_source.FitPositionsImage(
            name=point_source_dataset.name,
            positions=point_source_dataset.positions,
            noise_map=point_source_dataset.positions_noise_map,
            positions_solver=self.solver,
            tracer=tracer,
        )

    def fit_fluxes_for(self, point_source_dataset, tracer):

        return fit_point_source.FitFluxes(
            name=point_source_dataset.name,
            fluxes=point_source_dataset.fluxes,
            noise_map=point_source_dataset.fluxes_noise_map,
            positions=point_source_dataset.positions,
            tracer=tracer,
        )

    def visualize(self, paths, instance, during_analysis):

        tracer = self.tracer_for_instance(instance=instance)

        visualizer = vis.Visualizer(visualize_path=paths.image_path)

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return res.ResultPointSource(
            samples=samples, model=model, analysis=self, search=search
        )

    def save_attributes_for_aggregator(self, paths: af.Paths):

        with open(f"{paths.pickle_path}/dataset.pickle", "wb") as f:
            pickle.dump(self.point_source_dict, f)


class AttributesImaging(a.AttributesImaging):
    def __init__(
        self, cosmology, positions, hyper_model_image, hyper_galaxy_image_path_dict
    ):
        super().__init__(
            cosmology=cosmology,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        )

        self.positions = positions


class AttributesInterferometer(a.AttributesInterferometer):
    def __init__(
        self,
        cosmology,
        real_space_mask,
        positions,
        hyper_model_image,
        hyper_galaxy_image_path_dict,
    ):

        super().__init__(
            cosmology=cosmology,
            real_space_mask=real_space_mask,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        )

        self.positions = positions
