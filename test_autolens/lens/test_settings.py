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

        settings = al.SettingsLens(threshold=50.0)
        settings.resample_if_not_within_threshold(
            tracer=tracer, positions=al.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])
        )

        settings = al.SettingsLens(threshold=0.0)
        with pytest.raises(exc.RayTracingException):
            settings.resample_if_not_within_threshold(
                tracer=tracer, positions=al.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])
            )

        # No mass profile - doesnt raise exception

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
        )

        settings.resample_if_not_within_threshold(
            tracer=tracer, positions=al.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])
        )

        # Single plane - doesnt raise exception

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5, mass=al.mp.SphIsothermal())]
        )

        settings.resample_if_not_within_threshold(
            tracer=tracer, positions=al.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])
        )
