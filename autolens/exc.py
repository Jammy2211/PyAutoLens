import autofit as af



class CosmologyException(Exception):
    pass


class GalaxyException(Exception):
    pass


class RayTracingException(af.exc.FitException):
    pass


class PixelizationException(af.exc.FitException):
    pass


class InversionException(af.exc.FitException):
    pass


class FittingException(Exception):
    pass


class PlottingException(Exception):
    pass


class PhaseException(Exception):
    pass


class UnitsException(Exception):
    pass
