from astropy import cosmology as cosmo

import autofit as af

from autolens.analysis.analysis import AnalysisLensing
from autolens.analysis.visualizer import Visualizer
from autolens.point.point_dataset import PointDict
from autolens.point.fit_point.point_dict import FitPointDict
from autolens.point.model.result import ResultPoint

from autolens.point.point_solver import PointSolver
from autolens.analysis.settings import SettingsLens

from autolens import exc


class AnalysisPoint(af.Analysis, AnalysisLensing):
    def __init__(
        self,
        point_dict: PointDict,
        solver: PointSolver,
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

        try:
            fit = self.fit_positions_for(instance=instance)
            return fit.log_likelihood
        except ValueError as e:
            raise exc.FitException from e

    def fit_positions_for(self, instance):

        tracer = self.tracer_via_instance_from(instance=instance)

        return FitPointDict(
            point_dict=self.point_dict, tracer=tracer, point_solver=self.solver
        )

    def visualize(self, paths, instance, during_analysis):

        tracer = self.tracer_via_instance_from(instance=instance)

        visualizer = Visualizer(visualize_path=paths.image_path)

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return ResultPoint(samples=samples, model=model, analysis=self, search=search)

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        paths.save_object("dataset", self.point_dict)
