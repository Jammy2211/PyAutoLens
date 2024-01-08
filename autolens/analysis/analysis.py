import dill
import os
import json
import logging
import numpy as np
from os import path
from scipy.stats import norm
from typing import Dict, Optional, List, Union

from autoconf import conf
from autoconf.dictable import to_dict, output_to_json

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autogalaxy.analysis.analysis import AnalysisDataset as AgAnalysisDataset

from autolens.analysis.result import ResultDataset
from autolens.analysis.maker import FitMaker
from autolens.analysis.preloads import Preloads
from autolens.analysis.positions import PositionsLHResample
from autolens.analysis.positions import PositionsLHPenalty
from autolens.analysis.visualizer import Visualizer
from autolens.lens.ray_tracing import Tracer

from autolens.lens import ray_tracing_util

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisLensing:
    def __init__(
        self,
        positions_likelihood: Optional[
            Union[PositionsLHResample, PositionsLHPenalty]
        ] = None,
        cosmology: ag.cosmo.LensingCosmology = ag.cosmo.Planck15(),
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
        cosmology
            The Cosmology assumed for this analysis.
        """
        self.cosmology = cosmology
        self.positions_likelihood = positions_likelihood

    def tracer_via_instance_from(
        self,
        instance: af.ModelInstance,
        run_time_dict: Optional[Dict] = None,
        tracer_cls=Tracer,
    ) -> Tracer:
        """
        Create a `Tracer` from the galaxies contained in a model instance.

        If PyAutoFit's profiling tools are used with the analysis class, this function may receive a `run_time_dict`
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
        if hasattr(instance, "perturb"):
            instance.galaxies.subhalo = instance.perturb

        # TODO : Need to think about how we do this without building it into the model attribute names.
        # TODO : A Subhalo class that extends the Galaxy class maybe?

        if hasattr(instance.galaxies, "subhalo"):
            subhalo_centre = ray_tracing_util.grid_2d_at_redshift_from(
                galaxies=instance.galaxies,
                redshift=instance.galaxies.subhalo.redshift,
                grid=aa.Grid2DIrregular(values=[instance.galaxies.subhalo.mass.centre]),
                cosmology=self.cosmology,
            )

            instance.galaxies.subhalo.mass.centre = tuple(subhalo_centre.in_list[0])

        if hasattr(instance, "clumps"):
            return Tracer.from_galaxies(
                galaxies=instance.galaxies + instance.clumps,
                cosmology=self.cosmology,
                run_time_dict=run_time_dict,
            )

        if hasattr(instance, "cosmology"):
            cosmology = instance.cosmology
        else:
            cosmology = self.cosmology

        return tracer_cls.from_galaxies(
            galaxies=instance.galaxies,
            cosmology=cosmology,
            run_time_dict=run_time_dict,
        )

    def log_likelihood_positions_overwrite_from(
        self, instance: af.ModelInstance
    ) -> Optional[float]:
        """
        Call the positions overwrite log likelihood function, which add a penalty term to the likelihood if the
        positions of the multiple images of the lensed source do not trace close to one another in the
        source plane.

        This function handles a number of exceptions which may occur when calling the overwrite function via the
        `PositionsLikelihood` class, so that they do not need to be handled individually for each `Analysis` class.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        The penalty value of the positions log likelihood, if the positions do not trace close in the source plane,
        else a None is returned to indicate there is no penalty.
        """
        if self.positions_likelihood is not None:
            try:
                return self.positions_likelihood.log_likelihood_function_positions_overwrite(
                    instance=instance, analysis=self
                )
            except (ValueError, np.linalg.LinAlgError) as e:
                raise exc.FitException from e


class AnalysisDataset(AgAnalysisDataset, AnalysisLensing):
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
        Fits a lens model to a dataset via a non-linear search.

        This abstract Analysis class has attributes and methods for all model-fits which fit the model to a dataset
        (e.g. imaging or interferometer data).

        This class stores the Cosmology used for the analysis and settings that control aspects of the calculation,
        including how pixelizations, inversions and lensing calculations are performed.

        Parameters
        ----------
        dataset
            The imaging, interferometer or other dataset that the model if fitted too.
        positions_likelihood
            An object which alters the likelihood function to include a term which accounts for whether
            image-pixel coordinates in arc-seconds corresponding to the multiple images of the lensed source galaxy
            trace close to one another in the source-plane.
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        cosmology
            The AstroPy Cosmology assumed for this analysis.
        settings_inversion
            Settings controlling how an inversion is fitted during the model-fit, for example which linear algebra
            formalism is used.
        raise_inversion_positions_likelihood_exception
            If an inversion is used without the `positions_likelihood` it is likely a systematic solution will
            be inferred, in which case an Exception is raised before the model-fit begins to inform the user
            of this. This exception is not raised if this input is False, allowing the user to perform the model-fit
            anyway.
        """

        super().__init__(
            dataset=dataset,
            adapt_images=adapt_images,
            cosmology=cosmology,
            settings_inversion=settings_inversion,
        )

        AnalysisLensing.__init__(
            self=self,
            positions_likelihood=positions_likelihood,
            cosmology=cosmology,
        )

        self.preloads = self.preloads_cls()

        self.raise_inversion_positions_likelihood_exception = (
            raise_inversion_positions_likelihood_exception
        )

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            self.raise_inversion_positions_likelihood_exception = False

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        This function is called immediately before the non-linear search begins and performs final tasks and checks
        before it begins.

        This function:

        - Checks that the adapt-dataset is consistent with previous adapt-datasets if the model-fit is being
          resumed from a previous run.

        - Checks the model and raises exceptions if certain critieria are not met.

        Once inherited from it also visualizes objects which do not change throughout the model fit like the dataset.

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

        self.raise_exceptions(model=model)

    def raise_exceptions(self, model):
        has_pix = model.has_model(cls=(aa.Pixelization,)) or model.has_instance(
            cls=(aa.Pixelization,)
        )

        if has_pix:
            if (
                self.positions_likelihood is None
                and self.raise_inversion_positions_likelihood_exception
                and not conf.instance["general"]["test"][
                    "disable_positions_lh_inversion_check"
                ]
            ):
                raise exc.AnalysisException(
                    "You have begun a model-fit which reconstructs the source using a pixelization.\n\n"
                    "However, you have not input a `positions_likelihood` object.\n\n"
                    "It is likely your model-fit will infer an inaccurate solution.\n\n "
                    ""
                    "Please read the following readthedocs page for a description of why this is, and how to set up"
                    "a positions likelihood object:\n\n"
                    ""
                    "https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html"
                )

    @property
    def preloads_cls(self):
        return Preloads

    @property
    def fit_maker_cls(self):
        return FitMaker

    def save_results(self, paths: af.DirectoryPaths, result: ResultDataset):
        """
        At the end of a model-fit, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis it outputs the following:

        - The maximum log likelihood tracer of the fit.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        result
            The result of a model fit, including the non-linear search, samples and maximum likelihood tracer.
        """
        try:
            output_to_json(
                obj=result.max_log_likelihood_tracer,
                file_path=paths._files_path / "tracer.json",
            )
        except AttributeError:
            pass

        image_mesh_list = []

        for galaxy in result.instance.galaxies:
            pixelization_list = galaxy.cls_list_from(cls=aa.Pixelization)

            for pixelization in pixelization_list:
                if pixelization is not None:
                    image_mesh_list.append(pixelization.image_mesh)

        if len(image_mesh_list) > 0:
            paths.save_json(
                name="preload_mesh_grids_of_planes",
                object_dict=to_dict(
                    result.max_log_likelihood_fit.tracer_to_inversion.image_plane_mesh_grid_pg_list
                ),
            )
