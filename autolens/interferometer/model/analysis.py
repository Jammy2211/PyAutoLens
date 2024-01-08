import logging
import numpy as np
from typing import Dict, Optional, Tuple, Union

from autoconf.dictable import to_dict

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autoarray.exc import PixelizationException

from autolens.analysis.analysis import AnalysisDataset
from autolens.analysis.preloads import Preloads
from autolens.analysis.positions import PositionsLHResample
from autolens.analysis.positions import PositionsLHPenalty
from autolens.lens.ray_tracing import Tracer
from autolens.interferometer.model.result import ResultInterferometer
from autolens.interferometer.model.visualizer import VisualizerInterferometer
from autolens.interferometer.fit_interferometer import FitInterferometer

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisInterferometer(AnalysisDataset):
    def __init__(
        self,
        dataset,
        positions_likelihood: Optional[
            Union[PositionsLHResample, PositionsLHPenalty]
        ] = None,
        adapt_images: Optional[ag.AdaptImages] = None,
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
        settings_inversion: aa.SettingsInversion = None,
        raise_inversion_positions_likelihood_exception: bool = True,
    ):
        """
        Analysis classes are used by PyAutoFit to fit a model to a dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This Analysis class is used for all model-fits which fit galaxies (or objects containing galaxies like a
        `Tracer`) to an interferometer dataset.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and adapt images used for certain model
        classes.

        Parameters
        ----------
        dataset
            The interferometer dataset that the model is fitted too.
        positions_likelihood
            An object which alters the likelihood function to include a term which accounts for whether
            image-pixel coordinates in arc-seconds corresponding to the multiple images of the lensed source galaxy
            trace close to one another in the source-plane.
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        cosmology
            The Cosmology assumed for this analysis.
        settings_inversion
            Settings controlling how an inversion is fitted, for example which linear algebra formalism is used.
        raise_inversion_positions_likelihood_exception
            If an inversion is used without the `positions_likelihood` it is likely a systematic solution will
            be inferred, in which case an Exception is raised before the model-fit begins to inform the user
            of this. This exception is not raised if this input is False, allowing the user to perform the model-fit
            anyway.
        """
        super().__init__(
            dataset=dataset,
            positions_likelihood=positions_likelihood,
            adapt_images=adapt_images,
            cosmology=cosmology,
            settings_inversion=settings_inversion,
            raise_inversion_positions_likelihood_exception=raise_inversion_positions_likelihood_exception,
        )

    @property
    def interferometer(self):
        return self.dataset

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        This function is called immediately before the non-linear search begins and performs final tasks and checks
        before it begins.

        This function checks that the adapt-dataset is consistent with previous adapt-datasets if the model-fit is
        being resumed from a previous run, and it visualizes objects which do not change throughout the model fit
        like the dataset.

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

        if not paths.is_complete:
            logger.info(
                "PRELOADS - Setting up preloads, may take a few minutes for fits using an inversion."
            )

            self.set_preloads(paths=paths, model=model)

        return self

    def log_likelihood_function(self, instance):
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the interferometer dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) If the analysis has a adapt image, associated the model galaxy images of this dataset to the galaxies in
           the model instance.

        2) Extract attributes which model aspects of the data reductions, like the scaling the background sky
           and background noise.

        3) Extracts all galaxies from the model instance and set up a `Tracer`, which includes ordering the galaxies
           by redshift to set up each `Plane`.

        4) Use the `Tracer` and other attributes to create a `FitInterferometer` object, which performs steps such as
           creating model images of every galaxy in the plane, transforming them to the uv-plane via a Fourier transform
           and computing residuals, a chi-squared statistic and the log likelihood.

        Certain models will fail to fit the dataset and raise an exception. For example if an `Inversion` is used, the
        linear algebra calculation may be invalid and raise an Exception. In such circumstances the model is discarded
        and its likelihood value is passed to the non-linear search in a way that it ignores it (for example, using a
        value of -1.0e99).

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the interferometer data.
        """

        try:
            log_likelihood_positions_overwrite = (
                self.log_likelihood_positions_overwrite_from(instance=instance)
            )
            if log_likelihood_positions_overwrite is not None:
                return log_likelihood_positions_overwrite
        except Exception as e:
            raise e

        try:
            return self.fit_from(instance=instance).figure_of_merit
        except (
            PixelizationException,
            exc.PixelizationException,
            exc.InversionException,
            exc.GridException,
            exc.MeshException,
            ValueError,
            TypeError,
            np.linalg.LinAlgError,
            OverflowError,
        ) as e:
            raise exc.FitException from e

    def fit_from(
        self,
        instance: af.ModelInstance,
        preload_overwrite: Optional[Preloads] = None,
        run_time_dict: Optional[Dict] = None,
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
        preload_overwrite
            If a `Preload` object is input this is used instead of the preloads stored as an attribute in the analysis.
        check_positions
            Whether the multiple image positions of the lensed source should be checked, i.e. whether they trace
            within the position threshold of one another in the source plane.
        run_time_dict
            A dictionary which times functions called to fit the model to data, for profiling.

        Returns
        -------
        FitInterferometer
            The fit of the plane to the interferometer dataset, which includes the log likelihood.
        """

        tracer = self.tracer_via_instance_from(
            instance=instance, run_time_dict=run_time_dict
        )

        adapt_images = self.adapt_images_via_instance_from(instance=instance)

        preloads = self.preloads if preload_overwrite is None else preload_overwrite

        return FitInterferometer(
            dataset=self.dataset,
            tracer=tracer,
            adapt_images=adapt_images,
            settings_inversion=self.settings_inversion,
            preloads=preloads,
            run_time_dict=run_time_dict,
        )

    def visualize_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        PyAutoFit calls this function immediately before the non-linear search begins.

        It visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        visualizer = VisualizerInterferometer(visualize_path=paths.image_path)

        visualizer.visualize_interferometer(dataset=self.interferometer)

        if self.positions_likelihood is not None:
            visualizer.visualize_image_with_positions(
                image=self.dataset.dirty_image,
                positions=self.positions_likelihood.positions,
            )

        if self.adapt_images is not None:
            visualizer.visualize_adapt_images(adapt_images=self.adapt_images)

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis):
        """
        Outputs images of the maximum log likelihood model inferred by the model-fit. This function is called
        throughout the non-linear search at input intervals, and therefore provides on-the-fly visualization of how
        well the model-fit is going.

        The visualization performed by this function includes:

        - Images of the best-fit `Tracer`, including the images of each of its galaxies.

        - Images of the best-fit `FitInterferometer`, including the model-image, residuals and chi-squared of its fit
          to the imaging data.

        - The adapt-images of the model-fit showing how the galaxies are used to represent different galaxies in
          the dataset.

        - If adapt features are used to scale the noise, a `FitInterferometer` with these features turned off may be
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
        fit = self.fit_from(instance=instance)

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

    def make_result(
        self,
        samples: af.SamplesPDF,
    ):
        """
        After the non-linear search is complete create its `Result`, which includes:

        - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
          the maximum likelihood model, posteriors and other properties.

        - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
          an instance of the maximum log likelihood model).

        - The non-linear search used to perform the model fit.

        The `ResultInterferometer` object contains a number of methods which use the above objects to create the max
        log likelihood `Plane`, `FitInterferometer`, adapt-galaxy images,etc.

        Parameters
        ----------
        samples
            A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
            run of samples of the nested sampler.

        Returns
        -------
        ResultImaging
            The result of fitting the model to the imaging dataset, via a non-linear search.
        """
        return ResultInterferometer(samples=samples, analysis=self)

    def save_attributes(self, paths: af.DirectoryPaths):
        """
         Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `pickles`
         folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

         For this analysis, it uses the `AnalysisDataset` object's method to output the following:

         - The dataset's data.
         - The dataset's noise-map.
         - The settings associated with the dataset.
         - The settings associated with the inversion.
         - The settings associated with the pixelization.
         - The Cosmology.
         - The adapt image's model image and galaxy images, if used.

         This function also outputs attributes specific to an imaging dataset:

        - Its uv-wavelengths
        - Its real space mask.
        - The positions of the brightest pixels in the lensed source which are used to discard mass models.
        - The preloaded image-plane source plane pixelization if used by the analysis. This ensures that differences in
        the scikit-learn library do not lead to different pixelizations being computed if results are transferred from
        a HPC to laptop.

         It is common for these attributes to be loaded by many of the template aggregator functions given in the
         `aggregator` modules. For example, when using the database tools to perform a fit, the default behaviour is for
         the dataset, settings and other attributes necessary to perform the fit to be loaded via the pickle files
         output by this function.

         Parameters
         ----------
         paths
             The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
             visualization, and the pickled objects used by the aggregator output by this function.
        """
        super().save_attributes(paths=paths)

        hdu = aa.util.array_2d.hdu_for_output_from(
            array_2d=self.dataset.uv_wavelengths,
        )
        paths.save_fits(name="uv_wavelengths", hdu=hdu, prefix="dataset")
        paths.save_fits(
            name="real_space_mask",
            hdu=self.dataset.real_space_mask.hdu_for_output,
            prefix="dataset",
        )
        if self.positions_likelihood is not None:
            paths.save_json(
                name="positions",
                object_dict=to_dict(self.positions_likelihood.positions),
                prefix="dataset",
            )

    def profile_log_likelihood_function(
        self, instance: af.ModelInstance, paths: Optional[af.DirectoryPaths] = None
    ) -> Tuple[Dict, Dict]:
        """
        This function is optionally called throughout a model-fit to profile the log likelihood function.

        All function calls inside the `log_likelihood_function` that are decorated with the `profile_func` are timed
        with their times stored in a dictionary called the `run_time_dict`.

        An `info_dict` is also created which stores information on aspects of the model and dataset that dictate
        run times, so the profiled times can be interpreted with this context.

        The results of this profiling are then output to hard-disk in the `preloads` folder of the model-fit results,
        which they can be inspected to ensure run-times are as expected.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.

        Returns
        -------
        Two dictionaries, the profiling dictionary and info dictionary, which contain the profiling times of the
        `log_likelihood_function` and information on the model and dataset used to perform the profiling.
        """
        run_time_dict, info_dict = super().profile_log_likelihood_function(
            instance=instance,
        )

        info_dict["number_of_visibilities"] = self.dataset.visibilities.shape[0]
        info_dict["transformer_cls"] = self.dataset.transformer.__class__.__name__

        self.output_profiling_info(
            paths=paths, run_time_dict=run_time_dict, info_dict=info_dict
        )

        return run_time_dict, info_dict
