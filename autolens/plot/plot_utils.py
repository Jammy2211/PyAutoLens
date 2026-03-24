import numpy as np


def _to_lines(*items):
    """Convert multiple line sources into a flat list of (N, 2) numpy arrays."""
    result = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, list):
            for sub in item:
                try:
                    arr = np.array(sub.array if hasattr(sub, "array") else sub)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                        result.append(arr)
                except Exception:
                    pass
        else:
            try:
                arr = np.array(item.array if hasattr(item, "array") else item)
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                    result.append(arr)
            except Exception:
                pass
    return result or None


def _to_positions(*items):
    """Convert multiple position sources into a flat list of (N, 2) numpy arrays."""
    return _to_lines(*items)


def _critical_curves_from(tracer, grid):
    """Return (tangential_critical_curves, radial_critical_curves) as lists of arrays."""
    from autolens.lens import tracer_util

    try:
        tan_cc, rad_cc = tracer_util.critical_curves_from(tracer=tracer, grid=grid)
        return list(tan_cc), list(rad_cc)
    except Exception:
        return [], []


def _caustics_from(tracer, grid):
    """Return (tangential_caustics, radial_caustics) as lists of arrays."""
    from autolens.lens import tracer_util

    try:
        tan_ca, rad_ca = tracer_util.caustics_from(tracer=tracer, grid=grid)
        return list(tan_ca), list(rad_ca)
    except Exception:
        return [], []
