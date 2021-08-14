import autofit as af
from autofit.exc import *
from autoarray.exc import *


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


class AnalysisException(Exception):
    pass


class PreloadException(Exception):
    pass


class PointExtractionException(Exception):
    pass
