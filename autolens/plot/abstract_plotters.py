"""Compatibility shim — re-exports helpers used by subhalo.py and sensitivity.py."""
from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

from autogalaxy.plot.abstract_plotters import Plotter, _to_lines, _to_positions

__all__ = ["Plotter", "_to_lines", "_to_positions"]
