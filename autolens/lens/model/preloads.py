import logging
import numpy as np
from typing import Optional, List

import autoarray as aa

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class Preloads(aa.Preloads):
    def __init__(
        self,
        blurred_image: Optional[aa.Array2D] = None,
        sparse_grids_of_planes: Optional[aa.Grid2D] = None,
        mapper: Optional[aa.Mapper] = None,
        blurred_mapping_matrix: Optional[np.ndarray] = None,
        curvature_matrix_sparse_preload: Optional[np.ndarray] = None,
        curvature_matrix_preload_counts: Optional[np.ndarray] = None,
        w_tilde: Optional[aa.WTildeImaging] = None,
        use_w_tilde: Optional[bool] = None,
    ):
        """
        Class which offers a concise API for settings up the preloads, which before a model-fit are set up via
        a comparison of two fits using two different models. If a quantity in these two fits is identical, it does
        not change thoughout the model-fit and can therefore be preloaded to avoid computation, speeding up
        the analysis.

        For example, the image-plane source-plane pixelization grid (which may be computationally expensive to compute
        via a KMeans algorithm) does not change for the majority of model-fits, because the associated model parameters
        are fixed. Preloading avoids rerruning the KMeans algorithm for every model fitted, by preloading it in memory
        and using this preload in every fit.

        Parameters
        ----------
        blurred_image
            The preloaded array of values containing the blurred image of a lens model fit (e.g. that light profile of
            every galaxy in the model). This can be preloaded when no light profiles in the model vary.
        sparse_grids_of_planes
            The two dimension grid correspond to the sparse image plane grid in a lens fit, that is ray-traced to
            the source plane to form the source pixelization. This can be preloaded when no pixelizations in the model
            vary.
        mapper
            The mapper of a fit, which preloading avoids recalculation of the mapping matrix and image to source
            pixel mappings. This can be preloaded when no pixelizations in the model vary.
        blurred_mapping_matrix
            A matrix containing the mappings between PSF blurred image pixels and source pixels used in the linear
            algebra of an inversion. This can be preloaded when no mass profiles and pixelizations in the model vary.
        curvature_matrix_sparse_preload
            A matrix containing preloaded value used to construct the curvature matrix from the blurred mapping matrix.
            This can be preloaded when no mass profiles and pixelizations in the model vary.
        curvature_matrix_preload_counts
            A matrix containing the length of values in the curvature matrix preloaded, which are used to construct
            the curvature matrix from the blurred mapping matrix. This can be preloaded when no mass profiles and
            pixelizations in the model vary.
        w_tilde
            A class containing values that enable an inversion's linear algebra to use the w-tilde formalism. This can
            be preloaded when no component of the model changes the noise map (e.g. hyper galaxies are fixed).
        use_w_tilde
            Whether to use the w tilde formalism, which superseeds the value in `SettingsInversions` such that w tilde
            will be disabled for model-fits it is not applicable (e.g. because the noise-map changes).

        Returns
        -------
        Preloads
            The preloads object used to skip certain calculations in the log likelihood function.
        """
        super().__init__(
            sparse_grids_of_planes=sparse_grids_of_planes,
            mapper=mapper,
            blurred_mapping_matrix=blurred_mapping_matrix,
            curvature_matrix_sparse_preload=curvature_matrix_sparse_preload,
            curvature_matrix_preload_counts=curvature_matrix_preload_counts,
            w_tilde=w_tilde,
            use_w_tilde=use_w_tilde,
        )

        self.blurred_image = blurred_image

    def set_blurred_image(self, fit_0, fit_1):
        """
        If the `LightProfile`'s in a model are all fixed parameters their corresponding image and therefore PSF blurred
        image do not change during the model fit and can therefore be preloaded.

        This function compares the blurred image of two fit's corresponding to two model instances, and preloads
        the blurred image if the blurred image of both fits are the same.

        The preload is typically used though out search chaining pipelines, as it is common to fix the lens light for
        the majority of model-fits.

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """
        self.blurred_image = None

        if np.allclose(fit_0.blurred_image, fit_1.blurred_image):

            self.blurred_image = fit_0.blurred_image

            logger.info(
                "PRELOADS - Blurred image (e.g. the image of all light profiles) is preloaded for this model-fit."
            )

    def set_sparse_grid_of_planes(self, fit_0, fit_1):
        """
        If the `Pixelization`'s in a model are fixed their image-plane sparse grid (which defines the set of pixels
        that are ray-traced to construct the source-plane pixelization) do not change during the model=fit and
        can therefore be preloaded.

        This function compares the image plane sparse grid of two fit's corresponding to two model instances, and p
        reloads the grid if the grids of both fits are the same.

        The preload is typically used thoughout search chaining pipelines which use inversions, as it is common to
        for the pixelization's parameters to only vary in the hyper-searches.

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """

        self.sparse_grids_of_planes = None

        sparse_image_plane_grid_of_planes_0 = fit_0.tracer.sparse_image_plane_grids_of_planes_from_grid(
            grid=fit_0.dataset.grid_inversion
        )

        sparse_image_plane_grid_of_planes_1 = fit_1.tracer.sparse_image_plane_grids_of_planes_from_grid(
            grid=fit_1.dataset.grid_inversion
        )

        if sparse_image_plane_grid_of_planes_0[-1] is not None:

            if (
                sparse_image_plane_grid_of_planes_0[-1].shape[0]
                == sparse_image_plane_grid_of_planes_0[-1].shape[1]
            ):

                if np.allclose(
                    sparse_image_plane_grid_of_planes_0[-1],
                    sparse_image_plane_grid_of_planes_1[-1],
                ):

                    self.sparse_grids_of_planes = sparse_image_plane_grid_of_planes_0

                    logger.info(
                        "PRELOADS - Sparse grid of planes is preloaded for this model-fit."
                    )

    def set_mapper(self, fit_0, fit_1):
        """
        If the `MassProfile`'s and `Pixelization`'s in a model are fixed, the mapping of image-pixels to the
        source-pixels does not change during the model-fit and the `Mapper` containing this information can be
        preloaded. This includes preloading the `mapping_matrix`.

        This function compares the mapping matrix of two fit's corresponding to two model instances, and preloads the
        mapper if the mapping matrix of both fits are the same.

        The preload is typically used in searches where only light profiles vary (e.g. when only the lens's light is
        being fitted for).

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """

        self.mapper = None

        if fit_0.inversion is None:
            return

        mapper_0 = fit_0.inversion.mapper
        mapper_1 = fit_1.inversion.mapper

        if mapper_0.mapping_matrix.shape[1] == mapper_1.mapping_matrix.shape[1]:

            if np.allclose(mapper_0.mapping_matrix, mapper_1.mapping_matrix):

                self.mapper = mapper_0

                logger.info(
                    "PRELOADS - Mappers of planes preloaded for this model-fit."
                )

    def set_inversion(self, fit_0, fit_1):
        """
        If the `MassProfile`'s and `Pixelization`'s in a model are fixed, the mapping of image-pixels to the
        source-pixels does not change during the model-fit and matrices used to perform the linear algebra in an
        inversion can be preloaded, which help efficiently construct the curvature matrix.

        This function compares the blurred mapping matrix of two fit's corresponding to two model instances, and
        preloads the mapper if the mapping matrix of both fits are the same.

        The preload is typically used in searches where only light profiles vary (e.g. when only the lens's light is
        being fitted for).

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """

        self.blurred_mapping_matrix = None
        self.curvature_matrix_sparse_preload = None
        self.curvature_matrix_preload_counts = None

        inversion_0 = fit_0.inversion
        inversion_1 = fit_1.inversion

        if inversion_0 is None:
            return

        if (
            inversion_0.blurred_mapping_matrix.shape[1]
            == inversion_1.blurred_mapping_matrix.shape[1]
        ):

            if np.allclose(
                inversion_0.blurred_mapping_matrix, inversion_1.blurred_mapping_matrix
            ):

                self.blurred_mapping_matrix = inversion_0.blurred_mapping_matrix
                self.curvature_matrix_sparse_preload = (
                    inversion_0.curvature_matrix_sparse_preload
                )
                self.curvature_matrix_preload_counts = (
                    inversion_0.curvature_matrix_preload_counts
                )

                logger.info(
                    "PRELOADS - Inversion linear algebra quantities preloaded for this model-fit."
                )

    def set_w_tilde_imaging(self, fit_0, fit_1):
        """
        The w-tilde linear algebra formalism speeds up inversions by computing beforehand quantities that enable
        efficiently construction of the curvature matrix. These quantites can only be used if the noise-map is
        fixed, therefore this function preloads these w-tilde quantities if the noise-map does not change.

        This function compares the noise map of two fit's corresponding to two model instances, and preloads wtilde
        if the noise maps of both fits are the same.

        The preload is typically used through search chaining pipelines, as it is uncommon for the noise map to be
        scaled during the model-fit (although it is common for a fixed but scaled noise map to be used).

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """
        self.w_tilde = None
        self.use_w_tilde = False

        if fit_0.inversion is not None and np.allclose(
            fit_0.noise_map, fit_1.noise_map
        ):

            logger.info("PRELOADS - Computing W-Tilde... May take a moment.")

            preload, indexes, lengths = aa.util.inversion.w_tilde_curvature_preload_imaging_from(
                noise_map_native=fit_0.noise_map.native,
                kernel_native=fit_0.dataset.psf.native,
                native_index_for_slim_index=fit_0.dataset.mask._native_index_for_slim_index,
            )

            w_tilde = aa.WTildeImaging(
                curvature_preload=preload,
                indexes=indexes.astype("int"),
                lengths=lengths.astype("int"),
                noise_map_value=fit_0.noise_map[0],
            )

            self.w_tilde = w_tilde
            self.use_w_tilde = True

            logger.info("PRELOADS - W-Tilde preloaded for this model-fit.")

    @property
    def info(self) -> List[str]:
        """
        The information on what has or has not been preloaded, which is written to the file `preloads.summary`.

        Returns
        -------
            A list of strings containing True and False values as to whether a quantity has been preloaded.
        """
        line = [f"Blurred Image = {np.count_nonzero(self.blurred_image) != 0}\n"]
        line += [
            f"Sparse Grids of Planes = {self.sparse_grids_of_planes is not None}\n"
        ]
        line += [f"Mapper = {self.mapper is not None}\n"]
        line += [
            f"Blurred Mapping Matrix = {self.blurred_mapping_matrix is not None}\n"
        ]
        line += [
            f"Curvature Matrix Sparse Preload = {self.curvature_matrix_sparse_preload is not None}\n"
        ]
        line += [f"W Tilde = {self.w_tilde is not None}\n"]
        line += [f"Use W Tilde = {self.use_w_tilde}\n"]
        return line
