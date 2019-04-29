from autofit import exc


class ScaledArrayException(Exception):
    pass


class DataException(Exception):
    pass


class KernelException(Exception):
    pass


class MaskException(Exception):
    pass


class CosmologyException(Exception):
    pass


class GalaxyException(Exception):
    pass


class RayTracingException(exc.FitException):
    pass


class PixelizationException(exc.FitException):
    pass


class InversionException(exc.FitException):
    pass


class FittingException(Exception):
    pass


class PlottingException(Exception):
    pass


class PhaseException(Exception):
    pass


class UnitsException(Exception):
    pass