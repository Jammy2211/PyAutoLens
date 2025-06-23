import logging
import numpy as np
from typing import Dict, Optional, Tuple, Union

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autoarray.exc import PixelizationException

from autolens.analysis.analysis.dataset import AnalysisDataset
from autolens.analysis.positions import PositionsLH
from autolens.interferometer.model.result import ResultInterferometer
from autolens.interferometer.model.visualizer import VisualizerInterferometer
from autolens.interferometer.fit_interferometer import FitInterferometer

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisInterferometer(AnalysisDataset):
    Result = ResultInterferometer
    Visualizer = VisualizerInterferometer

    def __init__(
        self,
        dataset,
        positions_likelihood_list: Optional[PositionsLH] = None,
        adapt_image_maker: Optional[ag.AdaptImageMaker] = None,
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
        settings_inversion: aa.SettingsInversion = None,
        raise_inversion_positions_likelihood_exception: bool = True,
        title_prefix: str = None,
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
        positions_likelihood_list
            Alters the likelihood function to include a term which accounts for whether image-pixel coordinates in
            arc-seconds corresponding to the multiple images of each lensed source galaxy trace close to one another in
            their source-plane. This is a list, as it may support multiple planes, where a positions likelihood object
            is input for each plane (e.g. double source plane lensing).
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        cosmology
            The Cosmology assumed for this analysis.
        settings_inversion
            Settings controlling how an inversion is fitted, for example which linear algebra formalism is used.
        raise_inversion_positions_likelihood_exception
            If an inversion is used without the `positions_likelihood_list` it is likely a systematic solution will
            be inferred, in which case an Exception is raised before the model-fit begins to inform the user
            of this. This exception is not raised if this input is False, allowing the user to perform the model-fit
            anyway.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        super().__init__(
            dataset=dataset,
            positions_likelihood_list=positions_likelihood_list,
            adapt_image_maker=adapt_image_maker,
            cosmology=cosmology,
            settings_inversion=settings_inversion,
            raise_inversion_positions_likelihood_exception=raise_inversion_positions_likelihood_exception,
            title_prefix=title_prefix,
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
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """
        super().modify_before_fit(paths=paths, model=model)

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

        log_likelihood_penalty = self.log_likelihood_penalty_from(
            instance=instance
        )

        return self.fit_from(instance=instance).figure_of_merit + log_likelihood_penalty

    def fit_from(
        self,
        instance: af.ModelInstance,
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
            instance=instance,
        )

        adapt_images = self.adapt_images_via_instance_from(instance=instance)

        return FitInterferometer(
            dataset=self.dataset,
            tracer=tracer,
            adapt_images=adapt_images,
            settings_inversion=self.settings_inversion,
        )

    def save_attributes(self, paths: af.DirectoryPaths):
        """
         Before the model-fit begins, this routine saves attributes of the `Analysis` object to the `files` folder
         such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

         For this analysis, it uses the `AnalysisDataset` object's method to output the following:

         - The settings associated with the inversion.
         - The settings associated with the pixelization.
         - The Cosmology.
         - The adapt image's model image and galaxy images, as `adapt_images.fits`, if used.

         This function also outputs attributes specific to lens modeling:

        - The positions of the brightest pixels in the lensed source which are used to discard mass models.

        The following .fits files are also output via the plotter interface:

        - The real space mask applied to the dataset, in the `PrimaryHDU` of `dataset.fits`.
        - The interferometer dataset as `dataset.fits` (data / noise-map / uv_wavelengths).

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
        super().save_attributes(paths=paths)

        analysis = ag.AnalysisInterferometer(
            dataset=self.dataset,
        )

        analysis.save_attributes(paths=paths)
