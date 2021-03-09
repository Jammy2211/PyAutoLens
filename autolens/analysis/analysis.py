import numpy as np
import copy
from os import path
import pickle
from typing import List
import json
from astropy import cosmology as cosmo

from autoconf import conf
import autofit as af
from autofit.exc import FitException
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.inversion import pixelizations as pix, inversions as inv
from autoarray import preloads as pload
from autoarray.exc import PixelizationException, InversionException, GridException
from autogalaxy.galaxy import galaxy as g
from autogalaxy.analysis import analysis as a
from autolens.lens import ray_tracing
from autolens.fit import fit
from autolens.lens import settings
from autolens.analysis import result as res
from autolens.analysis import result_util
from autolens.analysis import visualizer as vis


class AnalysisDataset(a.AnalysisDataset):
    def __init__(
        self,
        dataset,
        positions: grid_2d_irregular.Grid2DIrregular = None,
        results=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        settings_lens=settings.SettingsLens(),
        preloads=pload.Preloads(),
    ):
        """

        Parameters
        ----------
        dataset
        positions : grid_2d_irregular.Grid2DIrregular
            Image-pixel coordinates in arc-seconds of bright regions of the lensed source that will map close to one
            another in the source-plane(s) for an accurate mass model, which can be used to discard unphysical mass
            models during model-fitting.
        results
        cosmology
        settings_pixelization
        settings_inversion
        settings_lens
        preloads
        """

        super().__init__(
            dataset=dataset,
            results=results,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
        )

        self.settings_lens = settings_lens

        self.positions = self.modify_positions(
            positions=positions,
            results=results,
            auto_positions_factor=self.settings_lens.auto_positions_factor,
        )

        self.settings_lens = self.modify_positions_threshold(results=results)

        self.settings_lens = self.modify_einstein_radius_estimate(results=results)

    def modify_positions(self, positions, results, auto_positions_factor):

        return result_util.updated_positions_from(
            positions=positions,
            results=results,
            auto_positions_factor=auto_positions_factor,
        )

    def modify_positions_threshold(self, results):

        positions_threshold = result_util.updated_positions_threshold_from(
            positions=self.positions,
            results=results,
            positions_threshold=self.settings_lens.positions_threshold,
            auto_positions_factor=self.settings_lens.auto_positions_factor,
            auto_positions_minimum_threshold=self.settings_lens.auto_positions_minimum_threshold,
        )

        return self.settings_lens.modify_positions_threshold(
            positions_threshold=positions_threshold
        )

    def modify_einstein_radius_estimate(self, results):
        """
        Use a previously estimated lens model to estimate the Einstein Radius of the system, and use this Einstein
        Radius as a resampling constrain in this `Analysis`'s model-fit, whereby mass models which do not produce this
        Einstein mass within a certain range of values are discarded.

        A typical use case is where we estimate an Einstein Mass by fitting a simple mass model (e.g. a  singular
        isothermal ellipsoid) which is sufficient to provide an accurate estimate of the Einstein Mass, which we can
        anticipate more complex mass models must reproduce. We then use this Einstein mass resampling to discard
        unphysical models when we fit a more complex mass model (e.g. a power-law mass profile).

        If the SettingsLens do not specify an `auto_einstein_radius_factor` or the previous results do not contain a
        best-fit mass model, the settings are returned unmodified.

        Parameters
        ----------
        settings_lens : SettingsLens
            A class containig all lens modeling settings.
        results : af.ResultsCollection
            A list of all previous model-fits performed before this Analysis.

        Returns
        -------
        SettingsLens
            Modified lens settings which include the previously estimated Einstein Radius value and fractional range
            outside of which unphysical models are discarded.
        """

        if (self.settings_lens.auto_einstein_radius_factor is not None) and (
            results is not None
        ):

            if results.last is not None:

                if results.last.max_log_likelihood_tracer.has_mass_profile:

                    einstein_radius = results.last.max_log_likelihood_tracer.einstein_radius_from_grid(
                        grid=self.dataset.data.mask.unmasked_grid_sub_1
                    )

                    return self.settings_lens.modify_einstein_radius_estimate(
                        einstein_radius_estimate=einstein_radius
                    )

        return self.settings_lens.modify_einstein_radius_estimate(
            einstein_radius_estimate=None
        )

    def tracer_for_instance(self, instance):

        if hasattr(instance, "perturbation"):
            instance.galaxies.subhalo = instance.perturbation

        return ray_tracing.Tracer.from_galaxies(
            galaxies=instance.galaxies, cosmology=self.cosmology
        )

    def stochastic_log_evidences_for_instance(self, instance) -> List[float]:
        raise NotImplementedError()

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

        self.settings_lens.check_einstein_radius_with_threshold_via_tracer(
            tracer=tracer, grid=self.dataset.grid
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

        return fit.FitImaging(
            masked_imaging=self.dataset,
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
                log_evidence = fit.FitImaging(
                    masked_imaging=self.dataset,
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

        visualizer.visualize_imaging(imaging=self.imaging.imaging)
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
        self, paths: af.Paths, samples: af.OptimizerSamples
    ):

        if conf.instance["general"]["hyper"]["stochastic_outputs"]:
            self.save_stochastic_outputs(paths=paths, samples=samples)

    def make_result(
        self,
        samples: af.PDFSamples,
        model: af.CollectionPriorModel,
        search: af.NonLinearSearch,
    ):
        return res.ResultImaging(
            samples=samples, model=model, analysis=self, search=search
        )


class AnalysisInterferometer(AnalysisDataset):
    def __init__(
        self,
        dataset,
        positions: grid_2d_irregular.Grid2DIrregular = None,
        results=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        settings_lens=settings.SettingsLens(),
        preloads=pload.Preloads(),
    ):

        super().__init__(
            dataset=dataset,
            positions=positions,
            results=results,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            settings_lens=settings_lens,
            preloads=preloads,
        )

        result = res.last_result_with_use_as_hyper_dataset(results=results)

        if result is not None:

            self.hyper_galaxy_visibilities_path_dict = (
                result.hyper_galaxy_visibilities_path_dict
            )

            self.hyper_model_visibilities = result.hyper_model_visibilities

        else:

            self.hyper_galaxy_visibilities_path_dict = None
            self.hyper_model_visibilities = None

    @property
    def interferometer(self):
        return self.dataset

    def log_likelihood_function(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the masked_interferometer in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model masked_interferometer itself
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
            fit = self.masked_interferometer_fit_for_tracer(
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
        Takes visibilities from the last result, if there is one, and associates them with galaxies in this phase
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

    def masked_interferometer_fit_for_tracer(
        self, tracer, hyper_background_noise, use_hyper_scalings=True
    ):

        return fit.FitInterferometer(
            masked_interferometer=self.dataset,
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
                log_evidence = fit.FitInterferometer(
                    masked_interferometer=self.dataset,
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

        fit = self.masked_interferometer_fit_for_tracer(
            tracer=tracer, hyper_background_noise=hyper_background_noise
        )

        visualizer = vis.Visualizer(visualize_path=paths.image_path)
        visualizer.visualize_interferometer(
            interferometer=self.interferometer.interferometer
        )
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

            fit = self.masked_interferometer_fit_for_tracer(
                tracer=tracer, hyper_background_noise=None, use_hyper_scalings=False
            )

            visualizer.visualize_fit_interferometer(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def save_results_for_aggregator(
        self, paths: af.Paths, samples: af.OptimizerSamples
    ):

        if conf.instance["general"]["hyper"]["stochastic_outputs"]:
            self.save_stochastic_outputs(paths=paths, samples=samples)

    def make_result(
        self,
        samples: af.PDFSamples,
        model: af.CollectionPriorModel,
        search: af.NonLinearSearch,
    ):
        return res.ResultInterferometer(
            samples=samples, model=model, analysis=self, search=search
        )
