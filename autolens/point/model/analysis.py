from typing import Dict, Optional

import autofit as af
import autogalaxy as ag

from autogalaxy.analysis.analysis.analysis import Analysis as AgAnalysis

from autolens.analysis.analysis.lens import AnalysisLens
from autolens.analysis.plotter_interface import PlotterInterface
from autolens.point.fit.positions.abstract import AbstractFitPositions
from autolens.point.fit.positions.image.pair_repeat import FitPositionsImagePairRepeat
from autolens.point.fit.dataset import FitPointDataset
from autolens.point.dataset import PointDataset
from autolens.point.model.result import ResultPoint
from autolens.point.solver import PointSolver

from autolens import exc

try:
    import numba

    NumbaException = numba.errors.TypingError
except ModuleNotFoundError:
    NumbaException = ValueError


class AnalysisPoint(AgAnalysis, AnalysisLens):
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
        dataset
            The imaging of the point-source dataset, which is not used for model-fitting but can be used for
            visualization.
        cosmology
            The cosmology of the ray-tracing calculation.
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
        Determine the fit of the strong lens system of lens galaxies and source galaxies to the point source data.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model masked_dataset itself
        """
        try:
            fit = self.fit_from(instance=instance)
            return fit.log_likelihood
        except (AttributeError, ValueError, TypeError, NumbaException) as e:
            raise exc.FitException from e

    def fit_from(
        self, instance, run_time_dict: Optional[Dict] = None
    ) -> FitPointDataset:
        tracer = self.tracer_via_instance_from(
            instance=instance, run_time_dict=run_time_dict
        )

        return FitPointDataset(
            dataset=self.dataset,
            tracer=tracer,
            solver=self.solver,
            fit_positions_cls=self.fit_positions_cls,
            run_time_dict=run_time_dict,
        )

    def visualize(self, paths, instance, during_analysis):
        tracer = self.tracer_via_instance_from(instance=instance)

        plotter_interface = PlotterInterface(image_path=paths.image_path)

    def save_attributes(self, paths: af.DirectoryPaths):
        ag.output_to_json(
            obj=self.dataset,
            file_path=paths._files_path / "dataset.json",
        )
