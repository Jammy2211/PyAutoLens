from astropy import cosmology as cosmo

import autofit as af
import autogalaxy as ag

from autogalaxy.quantity.model.visualizer import VisualizerQuantity

from autolens.analysis.visualizer import Visualizer
from autolens.analysis.analysis import AnalysisLensing
from autolens.analysis.settings import SettingsLens
from autogalaxy.quantity.plot.fit_quantity_plotters import FitQuantityPlotter
from autolens.quantity.model.result import ResultQuantity
from autolens.quantity.fit_quantity import FitQuantity


class AnalysisQuantity(ag.AnalysisQuantity, AnalysisLensing):
    def __init__(
        self,
        dataset: ag.DatasetQuantity,
        func_str: str,
        cosmology=cosmo.Planck15,
        settings_lens=SettingsLens(),
    ):
        """
        Analysis classes are used by PyAutoFit to fit a model to a dataset via a non-linear search.

        An Analysis class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data. The Analysis class handles many other tasks,
        such as visualization, outputting results to hard-disk and storing results in a format that can be loaded after
        the model-fit is complete using PyAutoFit's database tools.

        This Analysis class is used for model-fits which fit derived quantity of galaxies, for example their
        convergence, potential or deflection angles, to another model for that quantity. For example, one could find
        the `EllPowerLaw` mass profile model that best fits the deflection angles of an `EllNFW` mass profile.

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
        settings_lens
            Settings controlling the lens calculation, for example how close the lensed source's multiple images have
            to trace within one another in the source plane for the model to not be discarded.
        """
        super().__init__(dataset=dataset, func_str=func_str, cosmology=cosmology)

        AnalysisLensing.__init__(
            self=self, settings_lens=settings_lens, cosmology=cosmology
        )

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

    def visualize(
        self,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ) -> None:
        """
        Output images of the maximum log likelihood model inferred by the model-fit. This function is called throughout
        the non-linear search at regular intervals, and therefore provides on-the-fly visualization of how well the
        model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit `Plane`, including the images of each of its galaxies.

        - Images of the best-fit `FitQuantity`, including the model-image, residuals and chi-squared of its fit to
        the imaging data.

        The images output by this function are customized using the file `config/visualize/plots.ini`.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        during_analysis
            If True the visualization is being performed midway through the non-linear search before it is finished,
            which may change which images are output.
        """

        fit = self.fit_quantity_for_instance(instance=instance)

        visualizer = VisualizerQuantity(visualize_path=paths.image_path)
        visualizer.visualize_fit_quantity(
            fit=fit, fit_quanaity_plotter_cls=FitQuantityPlotter
        )

        visualizer = Visualizer(visualize_path=paths.image_path)
        visualizer.visualize_tracer(
            tracer=fit.tracer, grid=self.dataset.grid, during_analysis=during_analysis
        )

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ) -> ResultQuantity:
        """
        After the non-linear search is complete create its `ResultQuantity`, which includes:

        - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
        the maximum likelihood model, posteriors and other properties.

        - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
        an instance of the maximum log likelihood model).

        - The non-linear search used to perform the model fit.

        The `ResultQuantity` object contains a number of methods which use the above objects to create the max
        log likelihood `Tracer`, `FitQuantity`,etc.

        Parameters
        ----------
        samples
            A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
            run of samples of the nested sampler.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        search
            The non-linear search used to perform this model-fit.

        Returns
        -------
        ResultQuantity
            The result of fitting the model to the imaging dataset, via a non-linear search.
        """
        return ResultQuantity(
            samples=samples, model=model, analysis=self, search=search
        )
