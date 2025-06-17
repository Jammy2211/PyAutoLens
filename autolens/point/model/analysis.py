from typing import Dict, Optional

import autofit as af
import autogalaxy as ag

from autogalaxy.analysis.analysis.analysis import Analysis as AgAnalysis

from autolens.analysis.analysis.lens import AnalysisLens
from autolens.point.fit.positions.image.pair_repeat import FitPositionsImagePairRepeat
from autolens.point.fit.dataset import FitPointDataset
from autolens.point.dataset import PointDataset
from autolens.point.model.result import ResultPoint
from autolens.point.model.visualizer import VisualizerPoint
from autolens.point.solver import PointSolver

from autolens import exc

try:
    import numba

    NumbaException = numba.errors.TypingError
except ModuleNotFoundError:
    NumbaException = ValueError


class AnalysisPoint(AgAnalysis, AnalysisLens):
    Visualizer = VisualizerPoint
    Result = ResultPoint

    def __init__(
        self,
        dataset: PointDataset,
        solver: PointSolver,
        fit_positions_cls=FitPositionsImagePairRepeat,
        image=None,
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
        title_prefix: str = None,
    ):
        """
        Fits a lens model to a point source dataset (e.g. positions, fluxes, time delays) via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit lens models to point datasets, which may include some combination
        of positions, fluxes and time-delays.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and adapt images used for certain model
        classes.

        Parameters
        ----------
        dataset
            The `PointDataset` that is fitted by the model, which contains a combination of positions, fluxes and
            time-delays.
        solver
            Solves the lens equation in order to determine the image-plane positions of a point source by ray-tracing
            triangles to and from the source-plane.
        fit_positions_cls
            The class used to fit the positions of the point source dataset, which could be an image-plane or
            source-plane chi-squared.
        cosmology
            The Cosmology assumed for this analysis.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        super().__init__(cosmology=cosmology)

        AnalysisLens.__init__(self=self, cosmology=cosmology)

        self.dataset = dataset

        self.solver = solver
        self.fit_positions_cls = fit_positions_cls
        self.title_prefix = title_prefix

    def log_likelihood_function(self, instance):
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the point source dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) Extracts all galaxies from the model instance and set up a `Tracer`, which includes ordering the galaxies
           by redshift to set up each `Plane`.

        2) Use the `Tracer` and other attributes to create a `FitPointDataset` object, which performs the steps
           below to fit different components of the point source dataset.

        3) If the point source dataset has positions and model fits positions, perform this fit and compute the
           log likelihood. This calculation uses the `fit_positions_cls` object, which may be an image-plane or
           source-plane chi-squared.

        4) If the point source dataset has fluxes and model fits fluxes, perform this fit and compute the log likelihood.

        5) If the point source dataset has time-delays and model fits time-delays, perform this fit and compute the
           log likelihood [NOT SUPPORTED YET].

        6) Sum the log likelihoods of the positions, fluxes and time-delays (if they are fitted) to get the overall
           log likelihood of the model.

        Certain models will fail to fit the dataset and raise an exception. For example for ill defined mass models
        the `PointSolver` may find no solution. In such circumstances the model is discarded and its likelihood value
        is passed to the non-linear search in a way that it ignores it (for example, using a value of -1.0e99).

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the imaging data.
        """
        try:
            fit = self.fit_from(instance=instance)
            return fit.log_likelihood
        except (AttributeError, ValueError, TypeError, NumbaException) as e:
            print(e)
            dfdsfd
            raise exc.FitException from e

    def fit_from(
        self, instance, run_time_dict: Optional[Dict] = None
    ) -> FitPointDataset:
        tracer = self.tracer_via_instance_from(
            instance=instance, run_time_dict=run_time_dict
        )
        """
        Given a model instance create a `FitPointDataset` object.

        This function is used in the `log_likelihood_function` to fit the model to the imaging data and compute the
        log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        run_time_dict
            A dictionary which times functions called to fit the model to data, for profiling.

        Returns
        -------
        The fit of the lens model to the point source dataset.
        """
        return FitPointDataset(
            dataset=self.dataset,
            tracer=tracer,
            solver=self.solver,
            fit_positions_cls=self.fit_positions_cls,
            run_time_dict=run_time_dict,
        )

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis, it uses the `AnalysisDataset` object's method to output the following:

        - The dataset's point source dataset as a readable .json file.

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to perform a fit, the default behaviour is for
        the dataset, settings and other attributes necessary to perform the fit to be loaded via the pickle files
        output by this function.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        """
        ag.output_to_json(
            obj=self.dataset,
            file_path=paths._files_path / "dataset.json",
        )
