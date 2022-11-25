import os
import json
import logging
import numpy as np
from os import path
from scipy.stats import norm
from typing import Dict, Optional, List, Union

from autoconf import conf

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autogalaxy.analysis.analysis import AnalysisDataset as AgAnalysisDataset

from autolens.analysis.result import ResultDataset
from autolens.analysis.maker import FitMaker
from autolens.analysis.preloads import Preloads
from autolens.analysis.positions import PositionsLHResample
from autolens.analysis.positions import PositionsLHPenalty
from autolens.analysis.visualizer import Visualizer
from autolens.lens.ray_tracing import Tracer
from autolens.lens.to_inversion import TracerToInversion
from autolens.analysis.settings import SettingsLens

from autolens.lens import ray_tracing_util

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisLensing:
    def __init__(
        self,
        positions_likelihood: Optional[
            Union[PositionsLHResample, PositionsLHPenalty]
        ] = None,
        settings_lens: SettingsLens = SettingsLens(),
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
    ):
        """
        Analysis classes are used by PyAutoFit to fit a model to a dataset via a non-linear search.

        This abstract Analysis class has attributes and methods for all model-fits which include lensing calculations,
        but does not perform a model-fit by itself (and is therefore only inherited from).

        This class stores the Cosmology used for the analysis and settings that control specific aspects of the lensing
        calculation, for example how close the brightest pixels in the lensed source have to trace within one another
        in the source plane for the model to not be discarded.

        Parameters
        ----------
        settings_lens
            Settings controlling the lens calculation, for example how close the lensed source's multiple images have
            to trace within one another in the source plane for the model to not be discarded.
        cosmology
            The Cosmology assumed for this analysis.
        """
        self.cosmology = cosmology
        self.settings_lens = settings_lens or SettingsLens()
        self.positions_likelihood = positions_likelihood

    def tracer_via_instance_from(
        self, instance: af.ModelInstance, profiling_dict: Optional[Dict] = None
    ) -> Tracer:
        """
        Create a `Tracer` from the galaxies contained in a model instance.

        If PyAutoFit's profiling tools are used with the analsyis class, this function may receive a `profiling_dict`
        which times how long each set of the model-fit takes to perform.

        Parameters
        ----------
        instance
            An instance of the model that is fitted to the data by this analysis (whose parameters may have been set
            via a non-linear search).

        Returns
        -------
        Tracer
            An instance of the Tracer class that is used to then fit the dataset.
        """
        if hasattr(instance, "perturbation"):
            instance.galaxies.subhalo = instance.perturbation

        # TODO : Need to think about how we do this without building it into the model attribute names.
        # TODO : A Subhalo class that extends the Galaxy class maybe?

        if hasattr(instance.galaxies, "subhalo"):

            subhalo_centre = ray_tracing_util.grid_2d_at_redshift_from(
                galaxies=instance.galaxies,
                redshift=instance.galaxies.subhalo.redshift,
                grid=aa.Grid2DIrregular(grid=[instance.galaxies.subhalo.mass.centre]),
                cosmology=self.cosmology,
            )

            instance.galaxies.subhalo.mass.centre = tuple(subhalo_centre.in_list[0])

        if hasattr(instance, "clumps"):

            return Tracer.from_galaxies(
                galaxies=instance.galaxies + instance.clumps,
                cosmology=self.cosmology,
                profiling_dict=profiling_dict,
            )

        if hasattr(instance, "cosmology"):
            cosmology = instance.cosmology
        else:
            cosmology = self.cosmology

        return Tracer.from_galaxies(
            galaxies=instance.galaxies,
            cosmology=cosmology,
            profiling_dict=profiling_dict,
        )

    def log_likelihood_positions_overwrite_from(
        self, instance: af.ModelInstance
    ) -> Optional[float]:
        """
        Call the positions overwrite log likelihood function, which add a penalty term to the likelihood if the
        positions of the multiple images of the lensed source do not trace close to one another in the
        source plane.

        This function handles a number of exceptions which may occur when calling the overwrite function via the
        `PositionsLikelihood` class, so that they do not need to be handled individually for each `Analysis` class.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        The penalty value of the positions log likelihood, if the positions do not trace close in the source plane,
        else a None is returned to indicate there is no penalty.
        """
        if self.positions_likelihood is not None:

            try:
                return self.positions_likelihood.log_likelihood_function_positions_overwrite(
                    instance=instance, analysis=self
                )
            except (ValueError, np.linalg.LinAlgError) as e:
                raise exc.FitException from e


class AnalysisDataset(AgAnalysisDataset, AnalysisLensing):
    def __init__(
        self,
        dataset,
        positions_likelihood: Optional[
            Union[PositionsLHResample, PositionsLHPenalty]
        ] = None,
        hyper_dataset_result=None,
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
        settings_pixelization: aa.SettingsPixelization = None,
        settings_inversion: aa.SettingsInversion = None,
        settings_lens: SettingsLens = None,
        raise_inversion_positions_likelihood_exception: bool = True,
    ):
        """
        Analysis classes are used by PyAutoFit to fit a model to a dataset via a non-linear search.

        This abstract Analysis class has attributes and methods for all model-fits which fit the model to a dataset
        (e.g. imaging or interferometer data).

        This class stores the Cosmology used for the analysis and settings that control aspects of the calculation,
        including how pixelizations, inversions and lensing calculations are performed.

        Parameters
        ----------
        dataset
            The imaging, interferometer or other dataset that the model if fitted too.
        positions_likelihood
            An object which alters the likelihood function to include a term which accounts for whether
            image-pixel coordinates in arc-seconds corresponding to the multiple images of the lensed source galaxy
            trace close to one another in the source-plane.
        cosmology
            The AstroPy Cosmology assumed for this analysis.
        settings_pixelization
            settings controlling how a pixelization is fitted during the model-fit, for example if a border is used
            when creating the pixelization.
        settings_inversion
            Settings controlling how an inversion is fitted during the model-fit, for example which linear algebra
            formalism is used.
        settings_lens
            Settings controlling the lens calculation, for example how close the lensed source's multiple images have
            to trace within one another in the source plane for the model to not be discarded.
        raise_inversion_positions_likelihood_exception
            If an inversion is used without the `positions_likelihood` it is likely a systematic solution will
            be inferred, in which case an Exception is raised before the model-fit begins to inform the user
            of this. This exception is not raised if this input is False, allowing the user to perform the model-fit
            anyway.
        """

        super().__init__(
            dataset=dataset,
            hyper_dataset_result=hyper_dataset_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
        )

        AnalysisLensing.__init__(
            self=self,
            positions_likelihood=positions_likelihood,
            settings_lens=settings_lens,
            cosmology=cosmology,
        )

        self.settings_lens = settings_lens or SettingsLens()

        self.preloads = self.preloads_cls()

        self.raise_inversion_positions_likelihood_exception = (
            raise_inversion_positions_likelihood_exception
        )

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":

            self.raise_inversion_positions_likelihood_exception = False

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        PyAutoFit calls this function immediately before the non-linear search begins, therefore it can be used to
        perform tasks using the final model parameterization.

        This function:

        - Checks that the hyper-dataset is consistent with previous hyper-datasets if the model-fit is being
        resumed from a previous run.

        - Checks the model and raises exceptions if certain critieria are not met.

        Once inherited from it also visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        super().modify_before_fit(paths=paths, model=model)

        self.raise_exceptions(model=model)

    def raise_exceptions(self, model):

        if ag.util.model.has_pixelization_from(model=model):
            if (
                self.positions_likelihood is None
                and self.raise_inversion_positions_likelihood_exception
                and not conf.instance["general"]["test"][
                    "disable_positions_lh_inversion_check"
                ]
            ):
                raise exc.AnalysisException(
                    "You have begun a model-fit which reconstructs the source using a pixelization.\n\n"
                    "However, you have not input a `positions_likelihood` object.\n\n"
                    "It is likely your model-fit will infer an inaccurate solution.\n\n "
                    ""
                    "Please read the following readthedocs page for a description of why this is, and how to set up"
                    "a positions likelihood object:\n\n"
                    ""
                    "https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html"
                )

    @property
    def preloads_cls(self):
        return Preloads

    @property
    def fit_maker_cls(self):
        return FitMaker

    def log_likelihood_cap_from(
        self, stochastic_log_likelihoods_json_file: str
    ) -> float:
        """
        Certain `Inversion`'s have stochasticity in their log likelihood estimate (e.g. due to how different KMeans
        seeds change the pixelization constructed by a `VoronoiBrightnessImage` pixelization).

        A log likelihood cap can be applied to model-fits performed using these `Inversion`'s to improve error and
        posterior estimates. This log likelihood cap is estimated from a list of stochastic log likelihoods, where
        these log likelihoods are computed using the same model but with different KMeans seeds.

        This function computes the log likelihood cap of a model-fit by loading a set of stochastic log likelihoods
        from a .json file and fitting them with a 1D Gaussian. The cap is the mean value of this Gaussian.

        Parameters
        ----------
        stochastic_log_likelihoods_json_file
            A .json file which loads an ndarray of stochastic log likelihoods, which are likelihoods computed using the
            same model but with different KMeans seeds.

        Returns
        -------
        float
            A log likelihood cap which is applied in a stochastic model-fit to give improved error and posterior
            estimates.
        """
        try:
            with open(stochastic_log_likelihoods_json_file, "r") as f:
                stochastic_log_likelihoods = np.asarray(json.load(f))
        except FileNotFoundError:
            raise exc.AnalysisException(
                "The file 'stochastic_log_likelihoods.json' could not be found in the output of the model-fitting results"
                "in the analysis before the stochastic analysis. Rerun PyAutoLens with `stochastic_outputs=True` in the"
                "`general.ini` configuration file."
            )

        mean, sigma = norm.fit(stochastic_log_likelihoods)

        return mean

    def stochastic_log_likelihoods_via_instance_from(self, instance) -> List[float]:
        raise NotImplementedError()

    def save_results_for_aggregator(
        self, paths: af.DirectoryPaths, result: ResultDataset
    ):
        """
        At the end of a model-fit,  this routine saves attributes of the `Analysis` object to the `pickles`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis it outputs the following:

        - The stochastic log likelihoods of a pixelization, provided the pixelization has functionality that can
        compute likelihoods for different KMeans seeds and grids (e.g. `VoronoiBrightnessImage).

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        result
            The result of a lens model fit, including the non-linear search, samples and maximum likelihood tracer.
        """
        mesh_list = ag.util.model.mesh_list_from(model=result.model)

        if len(mesh_list) > 0:

            tracer_to_inversion = TracerToInversion(
                tracer=result.max_log_likelihood_tracer, dataset=self.dataset
            )

            sparse_image_plane_grid_pg_list = (
                tracer_to_inversion.sparse_image_plane_grid_pg_list
            )

            paths.save_object(
                "preload_sparse_grids_of_planes", sparse_image_plane_grid_pg_list
            )

        if conf.instance["general"]["hyper"]["stochastic_outputs"]:
            if len(mesh_list) > 0:
                for mesh in mesh_list:
                    if mesh.is_stochastic:
                        self.save_stochastic_outputs(
                            paths=paths, samples=result.samples
                        )

    def save_stochastic_outputs(self, paths: af.DirectoryPaths, samples: af.Samples):
        """
        Certain `Inversion`'s have stochasticity in their log likelihood estimate (e.g. due to how different KMeans
        seeds change the pixelization constructed by a `VoronoiBrightnessImage` pixelization).

        This function computes the stochastic log likelihoods of such a model, which are the log likelihoods computed
        using the same model but with different KMeans seeds.

        It outputs these stochastic likelihoods to a format which can be loaded via PyAutoFit's database tools, and
        may also be loaded if this analysis is extended with a stochastic model-fit that applies a log likelihood cap.

        This function also outputs visualization showing a histogram of the stochastic likelihood distribution.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        samples
            A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
            run of samples of the nested sampler.
        """
        stochastic_log_likelihoods_json_file = path.join(
            paths.output_path, "stochastic_log_likelihoods.json"
        )

        try:
            with open(stochastic_log_likelihoods_json_file, "r") as f:
                stochastic_log_likelihoods = np.asarray(json.load(f))
        except FileNotFoundError:
            instance = samples.max_log_likelihood()
            stochastic_log_likelihoods = (
                self.stochastic_log_likelihoods_via_instance_from(instance=instance)
            )

        if stochastic_log_likelihoods is None:
            return

        with open(stochastic_log_likelihoods_json_file, "w") as outfile:
            json.dump(
                [float(evidence) for evidence in stochastic_log_likelihoods], outfile
            )

        paths.save_object("stochastic_log_likelihoods", stochastic_log_likelihoods)

        visualizer = Visualizer(visualize_path=paths.image_path)

        visualizer.visualize_stochastic_histogram(
            stochastic_log_likelihoods=stochastic_log_likelihoods,
            max_log_evidence=np.max(samples.log_likelihood_list),
            histogram_bins=self.settings_lens.stochastic_histogram_bins,
        )
