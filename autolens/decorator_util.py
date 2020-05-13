from autoconf import conf
import numba

"""
Depending on if we're using a super computer, we want two different numba decorators:

If on laptop:

@numba.jit(nopython=True, cache=True, parallel=False)

If on super computer:

@numba.jit(nopython=True, cache=False, parallel=True)
"""

try:
    nopython = conf.instance.general.get("numba", "nopython", bool)
    cache = conf.instance.general.get("numba", "cache", bool)
    parallel = conf.instance.general.get("numba", "parallel", bool)
except Exception:
    nopython = True
    cache = True
    parallel = False


def jit(nopython=nopython, cache=cache, parallel=parallel):
    def wrapper(func):
        return numba.jit(func, nopython=nopython, cache=cache, parallel=parallel)

    return wrapper
