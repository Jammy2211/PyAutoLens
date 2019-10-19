import autofit as af


class RayTracingException(af.exc.FitException):
    pass


class FitException(Exception):
    pass


class PlottingException(Exception):
    pass


class PhaseException(Exception):
    pass