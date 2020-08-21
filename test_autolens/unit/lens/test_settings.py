import autolens as al
from autolens import exc

import pytest


class TestTags:
    def test__auto_positions_tag(self):

        settings = al.SettingsLens(
            auto_positions_factor=None, auto_positions_minimum_threshold=None
        )
        assert settings.auto_positions_factor_tag == ""
        settings = al.SettingsLens(
            auto_positions_factor=1.0, auto_positions_minimum_threshold=None
        )
        assert settings.auto_positions_factor_tag == "__auto_pos_x1.00"
        settings = al.SettingsLens(
            auto_positions_factor=2.56, auto_positions_minimum_threshold=None
        )
        assert settings.auto_positions_factor_tag == "__auto_pos_x2.56"
        settings = al.SettingsLens(
            auto_positions_factor=None, auto_positions_minimum_threshold=0.5
        )
        assert settings.auto_positions_factor_tag == ""
        settings = al.SettingsLens(
            auto_positions_factor=2.56, auto_positions_minimum_threshold=0.5
        )
        assert settings.auto_positions_factor_tag == "__auto_pos_x2.56_min_0.5"

    def test__positions_threshold_tag(self):

        settings = al.SettingsLens(positions_threshold=None)
        assert settings.positions_threshold_tag == ""
        settings = al.SettingsLens(positions_threshold=1.0)
        assert settings.positions_threshold_tag == "__pos_1.00"
        settings = al.SettingsLens(positions_threshold=2.56)
        assert settings.positions_threshold_tag == "__pos_2.56"


class TestCheckPositionsTrace:
    def test__positions_do_not_trace_within_threshold__raises_exception(self,):

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal()),
                al.Galaxy(redshift=1.0),
            ]
        )

        settings = al.SettingsLens(positions_threshold=50.0)
        settings.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer, positions=al.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)]])
        )

        settings = al.SettingsLens(positions_threshold=0.0)
        with pytest.raises(exc.RayTracingException):
            settings.check_positions_trace_within_threshold_via_tracer(
                tracer=tracer, positions=al.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)]])
            )

        # No mass profile - doesnt raise exception

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
        )

        settings.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer, positions=al.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)]])
        )

        # Single plane - doesnt raise exception

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal())]
        )

        settings.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer, positions=al.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)]])
        )
