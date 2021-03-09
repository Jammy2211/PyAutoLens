from . import aggregator as agg
from . import plot
from .dataset.imaging import MaskedImaging, SimulatorImaging
from .dataset.interferometer import MaskedInterferometer, SimulatorInterferometer
from .fit.fit import FitImaging, FitInterferometer
from .fit.fit_point_source import (
    FitPositionsSourceMaxSeparation,
    FitPositionsImage,
    FitFluxes,
)
from .lens.settings import SettingsLens
from .lens.ray_tracing import Tracer
from .lens.positions_solver import PositionsSolver
from .analysis.analysis import AnalysisImaging, AnalysisInterferometer
from .pipeline.setup import (
    SetupPipeline,
    SetupHyper,
    SetupSourceParametric,
    SetupSourceInversion,
    SetupMassTotal,
    SetupMassLightDark,
    SetupSubhalo,
)
from .pipeline.slam import (
    SLaMPipelineSourceParametric,
    SLaMPipelineSourceInversion,
    SLaMPipelineLightParametric,
    SLaMPipelineMass,
    SLaM,
)
from .pipeline.phase.settings import SettingsPhaseImaging
from .pipeline.phase.settings import SettingsPhaseInterferometer
from .pipeline.phase.settings import SettingsPhasePositions
from .pipeline.phase.point_source.phase import PhasePointSource

from autoconf import conf

conf.instance.register(__file__)

__version__ = "1.14.0"
