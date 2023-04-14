import logging
import os
from typing import Dict, Optional

import autofit as af
import autogalaxy as ag

from autolens.analysis.preloads import Preloads
from autolens.legacy.lens.ray_tracing import Tracer
from autolens.interferometer.model.result import ResultInterferometer
from autolens.interferometer.model.visualizer import VisualizerInterferometer
from autolens.legacy.interferometer.fit_interferometer import FitInterferometer

from autolens.interferometer.model.analysis import (
    AnalysisInterferometer as AnalysisInterferometerBase,
)

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisInterferometer(AnalysisInterferometerBase):
    def tracer_via_instance_from(
        self,
        instance: af.ModelInstance,
        profiling_dict: Optional[Dict] = None,
        tracer_cls=Tracer,
    ) -> Tracer:
        return super().tracer_via_instance_from(
            instance=instance, profiling_dict=profiling_dict, tracer_cls=tracer_cls
        )

    def fit_interferometer_via_instance_from(
        self,
        instance: af.ModelInstance,
        use_hyper_scaling: bool = True,
        preload_overwrite: Optional[Preloads] = None,
    ) -> FitInterferometer:
        """
        Given a model instance create a `FitInterferometer` object.

        This function is used in the `log_likelihood_function` to fit the model to the interferometer data and compute
        the log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        use_hyper_scaling
            If false, the scaling of the background sky and noise are not performed irrespective of the model components
            themselves.
        preload_overwrite
            If a `Preload` object is input this is used instead of the preloads stored as an attribute in the analysis.
        check_positions
            Whether the multiple image positions of the lensed source should be checked, i.e. whether they trace
            within the position threshold of one another in the source plane.
        profiling_dict
            A dictionary which times functions called to fit the model to data, for profiling.

        Returns
        -------
        FitInterferometer
            The fit of the plane to the interferometer dataset, which includes the log likelihood.
        """
        self.instance_with_associated_hyper_images_from(instance=instance)
        tracer = self.tracer_via_instance_from(instance=instance)

        hyper_background_noise = self.hyper_background_noise_via_instance_from(
            instance=instance
        )

        return self.fit_interferometer_via_tracer_from(
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scaling=use_hyper_scaling,
        )

    def fit_interferometer_via_tracer_from(
        self,
        tracer: Tracer,
        hyper_background_noise: Optional[ag.legacy.hyper_data.HyperBackgroundNoise],
        use_hyper_scaling: bool = True,
        preload_overwrite: Optional[Preloads] = None,
    ):
        """
        Given a `Tracer`, which the analysis constructs from a model instance, create a `FitInterferometer` object.

        This function is used in the `log_likelihood_function` to fit the model to the imaging data and compute the
        log likelihood.

        Parameters
        ----------
        tracer
            The tracer of galaxies whose ray-traced model images are used to fit the imaging data.
        hyper_image_sky
            A model component which scales the background sky level of the data before computing the log likelihood.
        hyper_background_noise
            A model component which scales the background noise level of the data before computing the log likelihood.
        use_hyper_scaling
            If false, the scaling of the background sky and noise are not performed irrespective of the model components
            themselves.
        preload_overwrite
            If a `Preload` object is input this is used instead of the preloads stored as an attribute in the analysis.
        profiling_dict
            A dictionary which times functions called to fit the model to data, for profiling.

        Returns
        -------
        FitImaging
            The fit of the plane to the imaging dataset, which includes the log likelihood.
        """
        preloads = self.preloads if preload_overwrite is None else preload_overwrite

        return FitInterferometer(
            dataset=self.dataset,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scaling=use_hyper_scaling,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            preloads=preloads,
        )

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis):
        """
        Outputs images of the maximum log likelihood model inferred by the model-fit. This function is called
        throughout the non-linear search at input intervals, and therefore provides on-the-fly visualization of how
        well the model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit `Tracer`, including the images of each of its galaxies.

        - Images of the best-fit `FitInterferometer`, including the model-image, residuals and chi-squared of its fit
          to the imaging data.

        - The hyper-images of the model-fit showing how the hyper galaxies are used to represent different galaxies in
          the dataset.

        - If hyper features are used to scale the noise, a `FitInterferometer` with these features turned off may be
          output, to indicate how much these features are altering the dataset.

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

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        instance = self.instance_with_associated_hyper_images_from(instance=instance)

        fit = self.fit_interferometer_via_instance_from(instance=instance)

        if self.positions_likelihood is not None:
            self.positions_likelihood.output_positions_info(
                output_path=paths.output_path, tracer=fit.tracer
            )

        if fit.inversion is not None:
            try:
                fit.inversion.reconstruction
            except exc.InversionException:
                return

        visualizer = VisualizerInterferometer(visualize_path=paths.image_path)

        try:
            visualizer.visualize_fit_interferometer(
                fit=fit, during_analysis=during_analysis
            )
        except exc.InversionException:
            pass

        tracer = fit.tracer_linear_light_profiles_to_light_profiles

        visualizer.visualize_tracer(
            tracer=tracer, grid=fit.grid, during_analysis=during_analysis
        )
        visualizer.visualize_galaxies(
            galaxies=tracer.galaxies, grid=fit.grid, during_analysis=during_analysis
        )
        if fit.inversion is not None:
            try:
                visualizer.visualize_inversion(
                    inversion=fit.inversion, during_analysis=during_analysis
                )
            except IndexError:
                pass

        visualizer.visualize_contribution_maps(tracer=fit.tracer)

        if visualizer.plot_fit_no_hyper:
            fit = self.fit_interferometer_via_tracer_from(
                tracer=fit.tracer,
                hyper_background_noise=None,
                use_hyper_scaling=False,
                preload_overwrite=Preloads(use_w_tilde=False),
            )

            try:
                visualizer.visualize_fit_interferometer(
                    fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
                )
            except exc.InversionException:
                pass

    def make_result(
        self,
        samples: af.SamplesPDF,
        model: af.Collection,
        sigma=1.0,
        use_errors=True,
        use_widths=False,
    ):
        """
        After the non-linear search is complete create its `Result`, which includes:

        - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
          the maximum likelihood model, posteriors and other properties.

        - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
          an instance of the maximum log likelihood model).

        - The non-linear search used to perform the model fit.

        The `ResultInterferometer` object contains a number of methods which use the above objects to create the max
        log likelihood `Plane`, `FitInterferometer`, hyper-galaxy images,etc.

        Parameters
        ----------
        samples
            A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
            run of samples of the nested sampler.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.

        Returns
        -------
        ResultImaging
            The result of fitting the model to the imaging dataset, via a non-linear search.
        """
        return ResultInterferometer(samples=samples, model=model, analysis=self)
