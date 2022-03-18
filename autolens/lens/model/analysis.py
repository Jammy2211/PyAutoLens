from astropy import cosmology as cosmo
import copy
import json
import logging
import numpy as np
import os
from os import path
from scipy.stats import norm
from typing import Dict, Optional, List

import autofit as af
import autoarray as aa

from autogalaxy.analysis.analysis import AnalysisDataset as AgAnalysisDataset

from autolens.lens.model.preloads import Preloads

from autolens import exc
from autolens.lens.model.maker import FitMaker
from autolens.lens.model.visualizer import Visualizer
from autolens.lens.ray_tracing import Tracer
from autolens.lens.model.settings import SettingsLens

from autolens.lens import ray_tracing_util

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisLensing:
    def __init__(
        self, settings_lens: SettingsLens = SettingsLens(), cosmology=cosmo.Planck15
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

            subhalo_centre = ray_tracing_util.grid_at_redshift_from(
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

        return Tracer.from_galaxies(
            galaxies=instance.galaxies,
            cosmology=self.cosmology,
            profiling_dict=profiling_dict,
        )


class AnalysisDataset(AgAnalysisDataset, AnalysisLensing):
    def __init__(
        self,
        dataset,
        positions: aa.Grid2DIrregular = None,
        hyper_dataset_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization: aa.SettingsPixelization = None,
        settings_inversion: aa.SettingsInversion = None,
        settings_lens: SettingsLens = None,
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
        positions
            Image-pixel coordinates in arc-seconds corresponding to the multiple images of the lensed source galaxy.
            If the `settings_lens` attribute has a `positions_threshold`, these positions must trace within this
            threshold of one another in the source-plane or else the model is discarded.
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
        """

        super().__init__(
            dataset=dataset,
            hyper_dataset_result=hyper_dataset_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
        )

        AnalysisLensing.__init__(
            self=self, settings_lens=settings_lens, cosmology=cosmology
        )

        self.positions = positions

        self.settings_lens = settings_lens or SettingsLens()

        self.preloads = Preloads()

    def set_preloads(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        It is common for the model to have components whose parameters are all fixed, and thus the way that component
        fits the data does not change. For example, if all parameter associated with the light profiles of galaxies
        in the model are fixed, the image generated from these galaxies will not change irrespective of the model
        parameters chosen by the non-linear search.

        Preloading exploits this to speed up the log likelihood function, by inspecting the model and storing in memory
        quantities that do not change. For the example above, the image of all galaxies would be stored in memory and
        to perform every fit in the `log_likelihood_funtion`.

        This function sets up all preload quantities, which are described fully in the `preloads` modules. This
        occurs directly before the non-linear search begins, to ensure the model parameterization is fixed.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        os.makedirs(paths.profile_path, exist_ok=True)

        fit_maker = FitMaker(model=model, fit_func=self.fit_func)

        fit_0 = fit_maker.fit_via_model_from(unit_value=0.45)
        fit_1 = fit_maker.fit_via_model_from(unit_value=0.55)

        if fit_0 is None or fit_1 is None:
            self.preloads = Preloads(failed=True)
        else:
            self.preloads = Preloads.setup_all_via_fits(fit_0=fit_0, fit_1=fit_1)
            self.preloads.check_via_fit(fit=fit_0)

        self.preloads.output_info_to_summary(file_path=paths.profile_path)

    def check_and_replace_hyper_images(self, paths: af.DirectoryPaths):
        """
        Using a the result of a previous model-fit, a hyper-dataset can be set up which adapts aspects of the model
        (e.g. the pixelization, regularization scheme) to the properties of the dataset being fitted.

        If the model-fit is being resumed from a previous run, this function checks that the model image and galaxy
        images used to set up the hyper-dataset are identical to those used previously. If they are not, it replaces
        them with the previous hyper image. This ensures consistency in the log likelihood function.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        """
        try:
            hyper_model_image = paths.load_object("hyper_model_image")

            if np.max(abs(hyper_model_image - self.hyper_model_image)) > 1e-8:

                logger.info(
                    "ANALYSIS - Hyper image loaded from pickle different to that set in Analysis class."
                    "Overwriting hyper images with values loaded from pickles."
                )

                self.hyper_model_image = hyper_model_image

                hyper_galaxy_image_path_dict = paths.load_object(
                    "hyper_galaxy_image_path_dict"
                )
                self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict

        except (FileNotFoundError, AttributeError, KeyError, ModuleNotFoundError):
            pass

    def modify_after_fit(
        self, paths: af.DirectoryPaths, model: af.AbstractPriorModel, result: af.Result
    ) -> "AnalysisDataset":
        """
        Call functions that perform tasks after a model-fit is completed, for example ensuring the figure of merit
        has not changed from previous estimates and resetting preloads.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        result
            The result of the model fit that has just been completed.
        """

        self.output_or_check_figure_of_merit_sanity(paths=paths, result=result)
        self.preloads.reset_all()

        return self

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
            instance = samples.max_log_likelihood_instance
            stochastic_log_likelihoods = self.stochastic_log_likelihoods_via_instance_from(
                instance=instance
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

    @property
    def no_positions(self):

        analysis = copy.deepcopy(self)

        analysis.positions = None
        analysis.settings_lens.positions_threshold = None

        return analysis

    @property
    def fit_func(self):
        raise NotImplementedError
