from autofit import exc


class ImagingException(Exception):
    pass


class KernelException(Exception):
    pass


class MaskException(Exception):
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
