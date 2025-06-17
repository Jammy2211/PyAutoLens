import jax.numpy as jnp
import logging
import numpy as np
from typing import Dict, List, Optional, Union

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autolens.analysis.positions import PositionsLH
from autolens.lens.tracer import Tracer

from autolens.lens import tracer_util

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisLens:
    def __init__(
        self,
        positions_likelihood_list: Optional[List[PositionsLH]] = None,
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
        self.positions_likelihood_list = positions_likelihood_list

    def tracer_via_instance_from(
        self,
        instance: af.ModelInstance,
        run_time_dict: Optional[Dict] = None,
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
            subhalo_centre = tracer_util.grid_2d_at_redshift_from(
                galaxies=instance.galaxies,
                redshift=instance.galaxies.subhalo.redshift,
                grid=aa.Grid2DIrregular(values=[instance.galaxies.subhalo.mass.centre]),
                cosmology=self.cosmology,
            )

            instance.galaxies.subhalo.mass.centre = tuple(subhalo_centre.in_list[0])

        if hasattr(instance, "cosmology"):
            cosmology = instance.cosmology
        else:
            cosmology = self.cosmology

        if hasattr(instance, "extra_galaxies"):
            if getattr(instance, "extra_galaxies", None) is not None:
                return Tracer(
                    galaxies=instance.galaxies + instance.extra_galaxies,
                    run_time_dict=run_time_dict,
                )

        return Tracer(
            galaxies=instance.galaxies,
            cosmology=cosmology,
            run_time_dict=run_time_dict,
        )

    def log_likelihood_penalty_from(
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
        log_likelihood_penalty = jnp.array(0.0)

        if self.positions_likelihood_list is not None:

            try:
                for positions_likelihood in self.positions_likelihood_list:
                    log_likelihood_penalty = (
                        positions_likelihood.log_likelihood_penalty_from(
                            instance=instance, analysis=self
                        )
                    )

                    log_likelihood_penalty += log_likelihood_penalty

                return log_likelihood_penalty
            except (ValueError, np.linalg.LinAlgError) as e:
                raise exc.FitException from e

        return log_likelihood_penalty
