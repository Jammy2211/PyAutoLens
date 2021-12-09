import autolens as al
from autolens import exc

import pytest


class TestCheckPositionsTrace:
    def test__positions_do_not_trace_within_threshold__raises_exception(self,):

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal()),
                al.Galaxy(redshift=1.0),
            ]
        )

        settings = al.SettingsLens(positions_threshold=50.0)
        settings.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer, positions=al.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])
        )

        settings = al.SettingsLens(positions_threshold=0.0)
        with pytest.raises(exc.RayTracingException):
            settings.check_positions_trace_within_threshold_via_tracer(
                tracer=tracer, positions=al.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])
            )

        # No mass profile - doesnt raise exception

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
        )

        settings.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer, positions=al.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])
        )

        # Single plane - doesnt raise exception

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal())]
        )

        settings.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer, positions=al.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])
        )
