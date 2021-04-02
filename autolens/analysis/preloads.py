import autofit as af

from autoarray.structures.grids.two_d import grid_2d
from autogalaxy.analysis import model_util
from autolens import exc


def preload_pixelization_grid_from(
    result: af.Result, model: af.Collection
) -> grid_2d.Grid2D:
    """
    If a model contains a `Pixelization` that is an `instance` whose parameters are fixed and the `Result` contains
    the grid of this pixelization corresponding to these parameters, the grid can be preloaded to avoid repeating
    calculations which recompute the pixelization grid every iteration of the log likelihood function.

    This function inspects the `Result` and `Model` and returns the appropriate pixelization grid for preloading,
    provideed it is suited to the analysis.

    Parameters
    ----------
    result
        The result containing the pixelization grid which is to be preloaded (corresponding to the maximum likelihood
        model of the model-fit).
    model
        The model, which is inspects to make sure the model-fit can have its pixelization preloaded.

    Returns
    -------
    Grid2D
        The (y,x) grid of coordinates representing the source plane pixelization centres.

    """

    if model_util.pixelization_from(model=model) is None:
        raise exc.PreloadException(
            "Cannot preload pixelization when the model does not include a pixelization"
        )

    if model_util.pixelization_is_model_from(model=model):
        raise exc.PreloadException(
            "Cannot preload pixelization when the model include a pixelization but it is a model"
            "component (preloading its grid will nullify changing its parameters)"
        )

    return result.max_log_likelihood_pixelization_grids_of_planes
