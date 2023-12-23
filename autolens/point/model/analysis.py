from typing import Callable, Dict, Optional, Tuple

import autofit as af
import autogalaxy as ag

from autogalaxy.analysis.analysis import Analysis as AgAnalysis

from autolens.analysis.analysis import AnalysisLensing
from autolens.analysis.visualizer import Visualizer
from autolens.point.point_dataset import PointDict
from autolens.point.fit_point.point_dict import FitPointDict
from autolens.point.model.result import ResultPoint

from autolens.point.point_solver import PointSolver

from autolens import exc

try:
    import numba

    NumbaException = numba.errors.TypingError
except ModuleNotFoundError:
    NumbaException = ValueError


class AnalysisPoint(AgAnalysis, AnalysisLensing):
    def __init__(
        self,
        point_dict: PointDict,
        solver: PointSolver,
        dataset=None,
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
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
        """

        super().__init__(cosmology=cosmology)

        AnalysisLensing.__init__(self=self, cosmology=cosmology)

        self.point_dict = point_dict

        self.solver = solver
        self.dataset = dataset

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

    def fit_from(self, instance, run_time_dict: Optional[Dict] = None) -> FitPointDict:
        tracer = self.tracer_via_instance_from(
            instance=instance, run_time_dict=run_time_dict
        )

        return FitPointDict(
            point_dict=self.point_dict,
            tracer=tracer,
            point_solver=self.solver,
            run_time_dict=run_time_dict,
        )

    def visualize(self, paths, instance, during_analysis):
        tracer = self.tracer_via_instance_from(instance=instance)

        visualizer = Visualizer(visualize_path=paths.image_path)

    def make_result(
        self,
        samples: af.SamplesPDF,
    ):
        return ResultPoint(samples=samples, analysis=self)

    def save_attributes(self, paths: af.DirectoryPaths):
        self.point_dict.output_to_json(
            file_path=paths._files_path / "point_dict.json", overwrite=True
        )
