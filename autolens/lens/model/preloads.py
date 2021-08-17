from typing import Optional

import autofit as af
import autoarray as aa
import autogalaxy as ag

from autolens import exc


class Preloads(aa.Preloads):
    @classmethod
    def setup(
        cls,
        result: af.Result,
        model: Optional[af.Collection] = None,
        pixelization: bool = False,
        inversion: bool = False,
        w_tilde: bool = False,
    ) -> aa.Preloads:
        """
        Class method which offers a concise API for settings up a preloads object, used throughout
        the `autolens_workspace` example scripts to make it explicit where a preload is being set up.

        Parameters
        ----------
        result
            The result containing the pixelization grid which is to be preloaded (corresponding to the maximum likelihood
            model of the model-fit).
        model
            The model, which is inspected to make sure the model-fit can have its pixelization preloaded.
        pixelization
            If `True` the pixelization grid of the source plane is preloaded.
        inversion
            If `True` certain matrices used in the linear algebra calculation of an inversion are preloaded.

        Returns
        -------
        Preloads
            The preloads object used to skip certain calculations in the log likelihood function.

        """
        if inversion:
            return preload_inversion_with_fixed_profiles(result=result, model=model)

        if pixelization:
            return preload_pixelization_grid_from(result=result, model=model)

        if w_tilde:
            return preload_w_tilde_from(result=result, model=model)


def preload_pixelization_grid_from(
    result: af.Result, model: Optional[af.Collection] = None
) -> aa.Preloads:
    """
    If a model contains a `Pixelization` that is an `instance` whose parameters are fixed and the `Result` contains
    the grid of this pixelization corresponding to these parameters, the grid can be preloaded to avoid repeating
    calculations which recompute the pixelization grid every iteration of the log likelihood function.

    This function inspects the `Result` and `Model` and returns a `Preloads` object with the appropriate pixelization
    grid for preloading. It raises an error if the `Model` is not suited to the preloading.

    Parameters
    ----------
    result
        The result containing the pixelization grid which is to be preloaded (corresponding to the maximum likelihood
        model of the model-fit).
    model
        The model, which is inspected to make sure the model-fit can have its pixelization preloaded.

    Returns
    -------
    Preloads
        The `Preloads` object containing the  (y,x) grid of coordinates representing the source plane pixelization
        centres.

    """

    if model is not None:

        if ag.util.model.pixelization_from(model=model) is None:
            raise exc.PreloadException(
                "Cannot preload pixelization when the model does not include a pixelization"
            )

        if ag.util.model.pixelization_is_model_from(model=model):
            raise exc.PreloadException(
                "Cannot preload pixelization when the model include a pixelization but it is a model"
                "component (preloading its grid will nullify changing its parameters)"
            )

    return aa.Preloads(
        sparse_grids_of_planes=result.max_log_likelihood_pixelization_grids_of_planes
    )


def preload_inversion_with_fixed_profiles(
    result: af.Result, model: Optional[af.Collection] = None
) -> aa.Preloads:
    """
    If the `MassProfile`'s in a model are all fixed parameters, and the parameters of the source `Pixelization` are
    also fixed, the mapping of image-pixels to the source-pixels does not change for every likelihood evaluations.
    Matrices used by the linear algebra calculation in an `Inversion` can therefore be preloaded.

    This function inspects the `Result` and `Model` and returns a `Preload` object with the correct quantities for p
    reloading. It raises an error if the `Model` is not suited to the preloading.

    The preload is typically used when the lens light is being fitted, and a fixed mass model and source pixelization
    and regularization are being used. This occurs in the LIGHT PIPELINE of the SLaM pipelines.

    Parameters
    ----------
    result
        The result containing the linear algebra matrices which are to be preloaded (corresponding to the maximum
        likelihood model of the model-fit).
    model
        The model, which is inspected to make sure the model-fit can have its `Inversion` quantities preloaded.

    Returns
    -------
    Grid2D
        The `Preloads` object containing the `Inversion` linear algebra matrices.
    """

    if model is not None:

        # if model.has_model(cls=mp.MassProfile):
        #     raise exc.PreloadException(
        #         "Cannot preload inversion when the mass profile is a model"
        #     )

        if ag.util.model.pixelization_is_model_from(model=model):
            raise exc.PreloadException(
                "Cannot preload inversion when the model includes a pixelization"
            )

    preloads = preload_pixelization_grid_from(result=result, model=model)
    inversion = result.max_log_likelihood_fit.inversion

    return aa.Preloads(
        sparse_grids_of_planes=preloads.sparse_grids_of_planes,
        blurred_mapping_matrix=inversion.blurred_mapping_matrix,
        curvature_matrix_sparse_preload=inversion.curvature_matrix_sparse_preload,
        curvature_matrix_preload_counts=inversion.curvature_matrix_preload_counts,
        mapper=inversion.mapper,
        use_w_tilde=False,
    )


def preload_w_tilde_from(
    result: af.Result, model: Optional[af.Collection] = None
) -> aa.Preloads:

    has_hyper_galaxy = ag.util.model.has_hyper_galaxy_from_model(model=model)

    # No hyper galaxy, so use of w_tilde with default image is fine.

    if not has_hyper_galaxy:

        return aa.Preloads()

    has_hyper_galaxy_model = ag.util.model.has_hyper_galaxy_model_from_model(
        model=model
    )

    # If there is a hyper galaxy in the model with free parameters, we cannot preload w_tilde. We use the preload
    # To bypass the w_tilde calculation.

    if has_hyper_galaxy_model:

        return aa.Preloads(use_w_tilde=False)

    # If there are hyper galaxies in the model but they all have fixed parameters (e.g. they are instances) we can
    # use w_tilde but we must modify it to use the modified noise map.

    has_hyper_galaxy_instance = ag.util.model.has_hyper_galaxy_instance_from_model(
        model=model
    )

    if has_hyper_galaxy_instance:

        dataset = result.max_log_likelihood_fit.dataset

        preload, indexes, lengths = aa.util.inversion.w_tilde_curvature_preload_imaging_from(
            noise_map_native=result.max_log_likelihood_fit.noise_map.native,
            kernel_native=dataset.psf.native,
            native_index_for_slim_index=dataset.mask._native_index_for_slim_index,
        )

        w_tilde = aa.WTildeImaging(
            curvature_preload=preload,
            indexes=indexes.astype("int"),
            lengths=lengths.astype("int"),
            noise_map_value=dataset.noise_map[0],
        )

        return aa.Preloads(w_tilde=w_tilde, use_w_tilde=True)
