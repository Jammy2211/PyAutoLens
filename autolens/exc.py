import autofit as af
from autofit.exc import *
from autoarray.exc import *
from autogalaxy.exc import *


class RayTracingException(af.exc.FitException):
    pass


class PositionsException(af.exc.FitException):
    pass


class PlottingException(Exception):
    pass


class PixelizationException(af.exc.FitException):
    pass


class SettingsException(Exception):
    pass


class PointExtractionException(Exception):
    pass
