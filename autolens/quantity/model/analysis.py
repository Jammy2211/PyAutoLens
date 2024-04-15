from typing import Optional

import autofit as af
import autogalaxy as ag

from autogalaxy.quantity.model.visualizer import VisualizerQuantity
from autolens.analysis.analysis.lens import AnalysisLens
from autolens.quantity.model.result import ResultQuantity
from autolens.quantity.fit_quantity import FitQuantity


class AnalysisQuantity(ag.AnalysisQuantity, AnalysisLens):
    Result = ResultQuantity
    Visualizer = VisualizerQuantity

    def __init__(
        self,
        dataset: ag.DatasetQuantity,
        func_str: str,
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
    ):
        """
        Analysis classes are used by PyAutoFit to fit a model to a dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit derived quantity of galaxies, for example their
        convergence, potential or deflection angles, to another model for that quantity. For example, one could find
        the `PowerLaw` mass profile model that best fits the deflection angles of an `NFW` mass profile.

        The `func_str` input defines what quantity is fitted, it corresponds to the function of the model `Plane`
        objects that is called to create the model quantity. For example, if `func_str="convergence_2d_from"`, the
        convergence is computed from each model `Plane`.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. the
        Cosmology used for the analysis).

        Parameters
        ----------
        dataset
            The `DatasetQuantity` dataset that the model is fitted too.
        func_str
            A string giving the name of the method of the input `Plane` used to compute the quantity that fits
            the dataset.
        cosmology
            The Cosmology assumed for this analysis.
        """
        super().__init__(dataset=dataset, func_str=func_str, cosmology=cosmology)

        AnalysisLens.__init__(self=self, cosmology=cosmology)

    def fit_quantity_for_instance(self, instance: af.ModelInstance) -> FitQuantity:
        """
        Given a model instance create a `FitImaging` object.

        This function is used in the `log_likelihood_function` to fit the model to the imaging data and compute the
        log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        FitQuantity
            The fit of the plane to the imaging dataset, which includes the log likelihood.
        """

        tracer = self.tracer_via_instance_from(instance=instance)

        return FitQuantity(dataset=self.dataset, tracer=tracer, func_str=self.func_str)
