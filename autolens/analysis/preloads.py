import logging
import numpy as np
from typing import Dict, Optional, List

import autoarray as aa
import autogalaxy as ag

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class Preloads(ag.Preloads):
    def __init__(
        self,
        w_tilde: Optional[aa.WTildeImaging] = None,
        use_w_tilde: Optional[bool] = None,
        blurred_image: Optional[aa.Array2D] = None,
        traced_grids_of_planes_for_inversion: Optional[aa.Grid2D] = None,
        image_plane_mesh_grid_pg_list: Optional[List[List[aa.Grid2D]]] = None,
        relocated_grid: Optional[aa.Grid2D] = None,
        mapper_list: Optional[aa.AbstractMapper] = None,
        mapper_galaxy_dict: Optional[Dict[aa.AbstractMapper, ag.Galaxy]] = None,
        operated_mapping_matrix: Optional[np.ndarray] = None,
        linear_func_operated_mapping_matrix_dict=None,
        data_linear_func_matrix_dict=None,
        mapper_operated_mapping_matrix_dict=None,
        regularization_matrix: Optional[np.ndarray] = None,
        log_det_regularization_matrix_term: Optional[float] = None,
        traced_mesh_grids_list_of_planes=None,
        image_plane_mesh_grid_list=None,
        failed=False,
    ):
        """
        Class which offers a concise API for settings up the preloads, which before a model-fit are set up via
        a comparison of two fits using two different models. If a quantity in these two fits is identical, it does
        not change throughout the model-fit and can therefore be preloaded to avoid computation, speeding up
        the analysis.

        For example, the image-plane source-plane pixelization grid (which may be computationally expensive to compute
        via a KMeans algorithm) does not change for the majority of model-fits, because the associated model parameters
        are fixed. Preloading avoids rerruning the KMeans algorithm for every model fitted, by preloading it in memory
        and using this preload in every fit.

        Parameters
        ----------
        blurred_image
            The preloaded array of values containing the blurred image of a model fit (e.g. that light profile of
            every galaxy in the model). This can be preloaded when no light profiles in the model vary.
        w_tilde
            A class containing values that enable an inversion's linear algebra to use the w-tilde formalism. This can
            be preloaded when no component of the model changes the noise map (e.g. galaxies are fixed).
        use_w_tilde
            Whether to use the w tilde formalism, which superseeds the value in `SettingsInversions` such that w tilde
            will be disabled for model-fits it is not applicable (e.g. because the noise-map changes).
        traced_grids_of_planes_for_inversion
            The two dimensional grids corresponding to the traced grids in a lens fit. This can be preloaded when no
             mass profiles in the model vary.
        image_plane_mesh_grid_pg_list
            The two dimensional grids corresponding to the sparse image plane grids in a lens fit, that is ray-traced to
            the source plane to form the source pixelization. This can be preloaded when no pixelizations in the model
            vary.
        relocated_grid
            The two dimensional grids corresponding to the grid that has had its border pixels relocated for a
            pixelization in a lens fit. This can be preloaded when no mass profiles in the model vary.
        mapper_list
            The mapper of a fit, which preloading avoids recalculation of the mapping matrix and image to source
            pixel mappings. This can be preloaded when no pixelizations in the model vary.
        operated_mapping_matrix
            A matrix containing the mappings between PSF blurred image pixels and source pixels used in the linear
            algebra of an inversion. This can be preloaded when no mass profiles and pixelizations in the model vary.

        Returns
        -------
        Preloads
            The preloads object used to skip certain calculations in the log likelihood function.
        """
        super().__init__(
            w_tilde=w_tilde,
            use_w_tilde=use_w_tilde,
            blurred_image=blurred_image,
            relocated_grid=relocated_grid,
            image_plane_mesh_grid_pg_list=image_plane_mesh_grid_pg_list,
            mapper_list=mapper_list,
            mapper_galaxy_dict=mapper_galaxy_dict,
            operated_mapping_matrix=operated_mapping_matrix,
            linear_func_operated_mapping_matrix_dict=linear_func_operated_mapping_matrix_dict,
            data_linear_func_matrix_dict=data_linear_func_matrix_dict,
            mapper_operated_mapping_matrix_dict=mapper_operated_mapping_matrix_dict,
            regularization_matrix=regularization_matrix,
            log_det_regularization_matrix_term=log_det_regularization_matrix_term,
            traced_mesh_grids_list_of_planes=traced_mesh_grids_list_of_planes,
            image_plane_mesh_grid_list=image_plane_mesh_grid_list,
        )

        self.traced_grids_of_planes_for_inversion = traced_grids_of_planes_for_inversion
        self.failed = failed

    @classmethod
    def setup_all_via_fits(cls, fit_0, fit_1) -> "Preloads":
        """
        Setup the Preloads from two fits which use two different lens model of a model-fit.

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.

        Returns
        -------
        Preloads
            Preloads which are set up based on the fit's passed in specific to a lens model.

        """

        preloads = cls()

        if isinstance(fit_0, aa.FitImaging):
            preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)
            preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)

        preloads.set_traced_grids_of_planes_for_inversion(fit_0=fit_0, fit_1=fit_1)
        preloads.set_image_plane_mesh_grid_pg_list(fit_0=fit_0, fit_1=fit_1)
        preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)
        preloads.set_mapper_list(fit_0=fit_0, fit_1=fit_1)

        if preloads.mapper_list is not None:
            preloads.mapper_galaxy_dict = fit_0.tracer_to_inversion.mapper_galaxy_dict

        preloads.set_operated_mapping_matrix_with_preloads(fit_0=fit_0, fit_1=fit_1)
        preloads.set_linear_func_inversion_dicts(fit_0=fit_0, fit_1=fit_1)
        preloads.set_curvature_matrix(fit_0=fit_0, fit_1=fit_1)
        preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

        return preloads

    def set_traced_grids_of_planes_for_inversion(self, fit_0, fit_1):
        """
        If the `MassProfiles`'s in a model are fixed their deflection angles and therefore corresponding traced grids
        do not change during the model-fit and can therefore be preloaded.

        This function compares the traced grids of two fit's corresponding to two model instances, and preloads the
        traced grids if the grids of both fits are the same. This preloaded grid is only used when constructing an
        inversion, because the `blurred_image` preload accounts for light profiles.

        The preload is typically used in adapt searches, where the mass model is fixed and the adapt-parameters are
        varied.

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """

        self.traced_grids_of_planes_for_inversion = None

        traced_grids_of_planes_0 = fit_0.tracer.traced_grid_2d_list_from(
            grid=fit_0.dataset.grid_pixelization
        )

        traced_grids_of_planes_1 = fit_1.tracer.traced_grid_2d_list_from(
            grid=fit_1.dataset.grid_pixelization
        )

        if traced_grids_of_planes_0[-1] is not None:
            if (
                traced_grids_of_planes_0[-1].shape[0]
                == traced_grids_of_planes_1[-1].shape[0]
            ):
                if (
                    np.max(
                        abs(traced_grids_of_planes_0[-1] - traced_grids_of_planes_1[-1])
                    )
                    < 1e-8
                ):
                    self.traced_grids_of_planes_for_inversion = traced_grids_of_planes_0

                    logger.info(
                        "PRELOADS - Traced grid of planes (for inversion) preloaded for this model-fit."
                    )

    def set_image_plane_mesh_grid_pg_list(self, fit_0, fit_1):
        """
        If the `Pixelization`'s in a model are fixed their image-plane sparse grid (which defines the set of pixels
        that are ray-traced to construct the source-plane pixelization) do not change during the model=fit and
        can therefore be preloaded.

        This function compares the image plane sparse grid of two fit's corresponding to two model instances, and p
        reloads the grid if the grids of both fits are the same.

        The preload is typically used throughout search chaining pipelines which use inversions, as it is common to
        for the pixelization's parameters to only vary in the adapt-searches.

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """

        self.image_plane_mesh_grid_pg_list = None

        image_plane_mesh_grid_pg_list_0 = (
            fit_0.tracer_to_inversion.image_plane_mesh_grid_pg_list
        )

        image_plane_mesh_grid_pg_list_1 = (
            fit_1.tracer_to_inversion.image_plane_mesh_grid_pg_list
        )

        if image_plane_mesh_grid_pg_list_0[-1] is not None:
            if image_plane_mesh_grid_pg_list_0[-1][0] is not None:
                if (
                    image_plane_mesh_grid_pg_list_0[-1][0].shape[0]
                    == image_plane_mesh_grid_pg_list_1[-1][0].shape[0]
                ):
                    if (
                        np.max(
                            abs(
                                image_plane_mesh_grid_pg_list_0[-1][0]
                                - image_plane_mesh_grid_pg_list_1[-1][0]
                            )
                        )
                        < 1e-8
                    ):
                        self.image_plane_mesh_grid_pg_list = (
                            image_plane_mesh_grid_pg_list_0
                        )

                        logger.info(
                            "PRELOADS - Sparse image-plane grids of planes is preloaded for this model-fit."
                        )

    @property
    def info(self) -> List[str]:
        """
        The information on what has or has not been preloaded, which is written to the file `preloads.summary`.

        Returns
        -------
            A list of strings containing True and False values as to whether a quantity has been preloaded.
        """
        line = [f"W Tilde = {self.w_tilde is not None}\n"]
        line += [f"Use W Tilde = {self.use_w_tilde}\n\n"]
        line += [f"Blurred Image = {np.count_nonzero(self.blurred_image) != 0}\n"]
        line += [
            f"Traced Grids of Planes (For LEq) = {self.traced_grids_of_planes_for_inversion is not None}\n"
        ]
        line += [
            f"Sparse Image-Plane Grids of Planes = {self.image_plane_mesh_grid_pg_list is not None}\n"
        ]
        line += [f"Relocated Grid = {self.relocated_grid is not None}\n"]
        line += [f"Mapper = {self.mapper_list is not None}\n"]
        line += [
            f"Blurred Mapping Matrix = {self.operated_mapping_matrix is not None}\n"
        ]
        line += [
            f"Inversion Linear Func (Linear Light Profile) Dicts = {self.linear_func_operated_mapping_matrix_dict is not None}\n"
        ]
        line += [f"Curvature Matrix = {self.curvature_matrix is not None}\n"]
        line += [
            f"Curvature Matrix Mapper Diag = {self.curvature_matrix_mapper_diag is not None}\n"
        ]
        line += [f"Regularization Matrix = {self.regularization_matrix is not None}\n"]
        line += [
            f"Log Det Regularization Matrix Term = {self.log_det_regularization_matrix_term is not None}\n"
        ]

        return line
