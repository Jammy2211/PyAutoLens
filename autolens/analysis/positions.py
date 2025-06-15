import numpy as np
from typing import Optional, Union
from os import path
import os
from typing import Dict

import autoarray as aa
import autofit as af

from autofit.tools.util import open_

import autogalaxy as ag

from autogalaxy.analysis.analysis.dataset import AnalysisDataset

from autolens.lens.tracer import Tracer
from autolens.point.max_separation import (
    SourceMaxSeparation,
)

from autolens import exc


class PositionsLH:
    def __init__(
        self,
        positions: aa.Grid2DIrregular,
        threshold: float,
        log_likelihood_penalty_factor: float = 1e8,
        plane_redshift: Optional[float] = None,
    ):
        """
        The `PositionsLH` objects add a penalty term to the likelihood of the **PyAutoLens** `log_likelihood_function`
        defined in the `Analysis` classes.

        The penalty term inspects the distance that the locations of the multiple images of the lensed source galaxy
        trace within one another in the source-plane and penalizes solutions where they trace far from one another,
        on the basis that this indicates an unphysical or inaccurate mass model. If they trace within the
        threshold the penalty term is not applied.

        For the `PositionsLH` object, if the multiple image coordinates do not trace within the source-plane
        threshold of one another a penalty to the likelihood is applied:

        `log_Likelihood_penalty_base - log_likelihood_penalty_factor * (max_source_plane_separation - threshold)`

        The penalty term reduces as the source-plane coordinates trace closer to one another, meaning that the
        initial stages of the non-linear search can sample mass models that reduce the threshold.

        For example, for one penalty term, if the multiple image coordinates are defined
        via `positions=aa.Grid2DIrregular([(1.0, 0.0), (-1.0, 0.0)]` and they do not trace within `threshold=0.3` of
        one another, the mass model will receive a large likelihood penalty.

        The default behaviour assumes a single lens plane and single source plane, meaning the input `positions`
        are the image-plane coordinates of one source galaxy at a specifc redshift.

        For multiple source planes, the `plane_redshift` can be used to pair the image-plane positions with the
        redshift of the source plane they trace too.

        Parameters
        ----------
        positions
            The arcsecond coordinates of the lensed source multiple images which are used to compute the likelihood
            penalty.
        threshold
            If the maximum separation of any two source plane coordinates is above the threshold the penalty term
            is applied.
        log_likelihood_penalty_factor
            A factor which multiplies how far source pixels do not trace within the threshold of one another, with a
            larger factor producing a larger penalty making the non-linear parameter space gradient steeper.
        plane_redshift
            The plane redshift of the lensed source multiple images, which is only required if position threshold
            for a double source plane lens system is being used where the specific plane is required.
        """

        try:
            mask = np.isfinite(positions.array).all(axis=1)
        except AttributeError:
            mask = np.isfinite(positions).all(axis=1)

        self.positions = aa.Grid2DIrregular(
            positions[mask]
        )
        self.threshold = threshold
        self.plane_redshift = plane_redshift

        if len(positions) == 1:
            raise exc.PositionsException(
                f"The positions input into the PositionsLikelihood object have length one "
                f"(e.g. it is only one (y,x) coordinate and therefore cannot be compared with other images).\n\n"
                "Please input more positions into the Positions."
            )

        self.log_likelihood_penalty_factor = log_likelihood_penalty_factor

    def output_positions_info(
        self, output_path: str, tracer: Tracer, overwrite_file: bool = True
    ):
        """
        Outputs a `positions.info` file which summarises the positions penalty term for a model fit, including:

        - The arc second coordinates of the lensed source multiple images used for the model-fit.
        - The radial distance of these coordinates from (0.0, 0.0).
        - The threshold value used by the likelihood penalty.
        - The maximum source plane separation of the maximum likelihood tracer.

        Parameters
        ----------
        output_path
        tracer

        Returns
        -------

        """

        flag = "w+" if overwrite_file else "a+"

        with open_(path.join(output_path, "positions.info"), flag) as f:

            positions_fit = SourceMaxSeparation(
                data=self.positions,
                noise_map=None,
                tracer=tracer,
                plane_redshift=self.plane_redshift,
            )

            distances = positions_fit.data.distances_to_coordinate_from(
                coordinate=(0.0, 0.0)
            )

            if self.plane_redshift is None:
                f.write(f"Plane Index: -1 \n")
            else:
                f.write(f"Plane Redshift: {self.plane_redshift} \n")

            f.write(f"Positions: \n {self.positions} \n\n")
            f.write(f"Radial Distance from (0.0, 0.0): \n {distances} \n\n")
            f.write(f"Threshold = {self.threshold} \n")
            f.write(
                f"Max Source Plane Separation of Maximum Likelihood Model = {positions_fit.max_separation_of_plane_positions}"
            )
            f.write("")

    def log_likelihood_penalty_base_from(
        self, dataset: Union[aa.Imaging, aa.Interferometer]
    ) -> float:
        """
        The fast log likelihood penalty scheme returns an alternative penalty log likelihood for any model where the
        image-plane positions do not trace within a threshold distance of one another in the source-plane.

        This `log_likelihood_penalty` is defined as:

        log_Likelihood_penalty_base - log_likelihood_penalty_factor * (max_source_plane_separation - threshold)

        The `log_likelihood_penalty` is only used if `max_source_plane_separation > threshold`.

        This function returns the `log_likelihood_penalty_base`, which represents the lowest possible likelihood
        solutions a model-fit can give. It is the chi-squared of model-data consisting of all zeros plus
        the noise normalziation term.

        Parameters
        ----------
        dataset
            The imaging or interferometer dataset from which the penalty base is computed.
        """

        residual_map = aa.util.fit.residual_map_from(
            data=dataset.data, model_data=np.zeros(dataset.data.shape)
        )

        if isinstance(dataset, aa.Imaging):
            chi_squared_map = aa.util.fit.chi_squared_map_from(
                residual_map=residual_map, noise_map=dataset.noise_map
            )

            chi_squared = aa.util.fit.chi_squared_from(
                chi_squared_map=chi_squared_map.array
            )

            noise_normalization = aa.util.fit.noise_normalization_from(
                noise_map=dataset.noise_map.array
            )

        else:
            chi_squared_map = aa.util.fit.chi_squared_map_complex_from(
                residual_map=residual_map, noise_map=dataset.noise_map
            )

            chi_squared = aa.util.fit.chi_squared_complex_from(
                chi_squared_map=chi_squared_map
            )

            noise_normalization = aa.util.fit.noise_normalization_complex_from(
                noise_map=dataset.noise_map
            )

        return -0.5 * (chi_squared + noise_normalization)

    def log_likelihood_penalty_from(self, tracer: Tracer) -> Optional[float]:
        """
        The fast log likelihood penalty scheme returns an alternative penalty log likelihood for any model where the
        image-plane positions to not trace within a threshold distance of one another in the source-plane.

        This `log_likelihood_penalty` is defined as:

        log_Likelihood_penalty_base - log_likelihood_penalty_factor * (max_source_plane_separation - threshold)

        The `log_likelihood_penalty` is only used if `max_source_plane_separation > threshold`.

        Parameters
        ----------
        dataset
            The imaging or interferometer dataset from which the penalty base is computed.
        """
        if not tracer.has(cls=ag.mp.MassProfile) or len(tracer.planes) == 1:
            return

        positions_fit = SourceMaxSeparation(
            data=self.positions,
            noise_map=None,
            tracer=tracer,
            plane_redshift=self.plane_redshift,
        )

        if not positions_fit.max_separation_within_threshold(self.threshold):

            return self.log_likelihood_penalty_factor * (
                positions_fit.max_separation_of_plane_positions - self.threshold
            )

    def log_likelihood_function_positions_overwrite(
        self, instance: af.ModelInstance, analysis: AnalysisDataset
    ) -> Optional[float]:
        """
        This is called in the `log_likelihood_function` of certain `Analysis` classes to add the penalty term of
        this class, which penalies mass models which do not trace within the threshold of one another in the
        source-plane.

        Parameters
        ----------
        instance
            The instance of the lens model that is being fitted for this iteration of the non-linear search.
        analysis
            The analysis class from which the log likliehood function is called.
        """
        tracer = analysis.tracer_via_instance_from(instance=instance)

        if not tracer.has(cls=ag.mp.MassProfile) or len(tracer.planes) == 1:
            return

        log_likelihood_positions_penalty = self.log_likelihood_penalty_from(
            tracer=tracer
        )

        if log_likelihood_positions_penalty is None:
            return None

        log_likelihood_penalty_base = self.log_likelihood_penalty_base_from(
            dataset=analysis.dataset
        )

        return log_likelihood_penalty_base - log_likelihood_positions_penalty
