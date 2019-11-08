import autofit as af


class RayTracingException(af.exc.FitException):
    pass


class PlottingException(Exception):
    pass


class PhaseException(Exception):
    pass


class PixelizationException(af.exc.FitException):
    pass
