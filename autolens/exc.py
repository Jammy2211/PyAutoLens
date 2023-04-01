import autofit as af
from autofit.exc import *
from autoarray.exc import *
from autogalaxy.exc import *


class RayTracingException(af.exc.FitException):
    """
    Raises exceptions associated with the `lens/ray_tracing.py` module and `Tracer` class.

    For example if the multiple image positions do not trace without a threshold of one another, in order to
    resample inaccurate mass models during a model-fit.

    This exception inehrits from a `FitException`. This means that if this exception is raised during a model-fit in
    the analysis class's `log_likelihood_function` that model is resampled and does not terminate the code.
    """

    pass


class PositionsException(af.exc.FitException):
    """
    Raises exceptions associated with the positions data in the  `point` module.

    For example if the multiple image positions do not meet certain format requirements.

    This exception inehrits from a `FitException`. This means that if this exception is raised during a model-fit in
    the analysis class's `log_likelihood_function` that model is resampled and does not terminate the code.
    """

    pass


class PixelizationException(af.exc.FitException):
    """
    Raises exceptions associated with the `inversion/pixelization` modules and `Pixelization` classes.

    For example if a `Rectangular` mesh has dimensions below 3x3.

    This exception overwrites `autoarray.exc.PixelizationException` in order to add a `FitException`. This means that
    if this exception is raised during a model-fit in the analysis class's `log_likelihood_function` that model
    is resampled and does not terminate the code.
    """

    pass


class PointExtractionException(Exception):
    """
    Raises exceptions associated with the extraction of quantities in the  `point` module, where the name of a
    `PointSource` profile often relates to a model-component.

    For example if one tries to extract a profile `point_1` but there is no corresponding `PointSource` profile
    named `point_1`.
    """

    pass
