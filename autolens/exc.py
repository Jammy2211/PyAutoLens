class ImagingException(Exception):
    pass


class KernelException(Exception):
    pass


class PriorException(Exception):
    pass


class MultiNestException(Exception):
    pass


class MaskException(Exception):
    pass


class CoordinatesException(Exception):
    """Exception thrown when coordinate assertion fails"""
    pass


class PixelizationException(Exception):
    pass


class ReconstructionException(Exception):
    pass


class RayTracingException(Exception):
    pass


class PhaseException(Exception):
    pass
