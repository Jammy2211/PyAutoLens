import autolens as al
from autolens import exc

import pytest


class TestTags:
    def test__positions_threshold_tag(self):

        settings = al.SettingsLens(positions_threshold=None)
        assert settings.positions_threshold_tag == "pos_off"
        settings = al.SettingsLens(positions_threshold=1.0)
        assert settings.positions_threshold_tag == "pos_on"

    def test__stochastic_likelihood_resamples_tag(self):

        settings = al.SettingsLens(stochastic_likelihood_resamples=None)
        assert settings.stochastic_likelihood_resamples_tag == ""
        settings = al.SettingsLens(stochastic_likelihood_resamples=2)
        assert settings.stochastic_likelihood_resamples_tag == "__lh_resamples_2"
        settings = al.SettingsLens(stochastic_likelihood_resamples=3)
        assert settings.stochastic_likelihood_resamples_tag == "__lh_resamples_3"

    def test__tag(self):

        settings = al.SettingsLens(
            positions_threshold=1.0,
            auto_positions_factor=2.56,
            auto_positions_minimum_threshold=0.5,
        )
        assert settings.tag == "lens[pos_on]"

        settings = al.SettingsLens(
            positions_threshold=1.0,
            auto_positions_factor=2.56,
            auto_positions_minimum_threshold=0.5,
            stochastic_likelihood_resamples=2,
        )
        assert settings.tag == "lens[pos_on__lh_resamples_2]"


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
            tracer=tracer,
            positions=al.Grid2DIrregularGrouped([[(1.0, 1.0), (2.0, 2.0)]]),
        )

        settings = al.SettingsLens(positions_threshold=0.0)
        with pytest.raises(exc.RayTracingException):
            settings.check_positions_trace_within_threshold_via_tracer(
                tracer=tracer,
                positions=al.Grid2DIrregularGrouped([[(1.0, 1.0), (2.0, 2.0)]]),
            )

        # No mass profile - doesnt raise exception

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5), al.Galaxy(redshift=1.0)]
        )

        settings.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer,
            positions=al.Grid2DIrregularGrouped([[(1.0, 1.0), (2.0, 2.0)]]),
        )

        # Single plane - doesnt raise exception

        tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5, mass=al.mp.SphericalIsothermal())]
        )

        settings.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer,
            positions=al.Grid2DIrregularGrouped([[(1.0, 1.0), (2.0, 2.0)]]),
        )


class TestCheckEinsteinRadius:
    def test__einstein_radius_outside_auto_range__raises_exception(self):

        grid = al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.2)

        settings = al.SettingsLens(auto_einstein_radius_factor=0.2)
        settings = settings.modify_einstein_radius_estimate(
            einstein_radius_estimate=1.0
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=1.0)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        settings.check_einstein_radius_with_threshold_via_tracer(
            tracer=tracer, grid=grid
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=0.81)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        settings.check_einstein_radius_with_threshold_via_tracer(
            tracer=tracer, grid=grid
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=1.19)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        settings.check_einstein_radius_with_threshold_via_tracer(
            tracer=tracer, grid=grid
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=1.21)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        with pytest.raises(exc.RayTracingException):
            settings.check_einstein_radius_with_threshold_via_tracer(
                tracer=tracer, grid=grid
            )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=0.79)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        with pytest.raises(exc.RayTracingException):
            settings.check_einstein_radius_with_threshold_via_tracer(
                tracer=tracer, grid=grid
            )

        settings = al.SettingsLens(auto_einstein_radius_factor=0.5)
        settings = settings.modify_einstein_radius_estimate(
            einstein_radius_estimate=2.0
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=1.01)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        settings.check_einstein_radius_with_threshold_via_tracer(
            tracer=tracer, grid=grid
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=2.99)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        settings.check_einstein_radius_with_threshold_via_tracer(
            tracer=tracer, grid=grid
        )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=0.99)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        with pytest.raises(exc.RayTracingException):
            settings.check_einstein_radius_with_threshold_via_tracer(
                tracer=tracer, grid=grid
            )

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=3.02)
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        with pytest.raises(exc.RayTracingException):
            settings.check_einstein_radius_with_threshold_via_tracer(
                tracer=tracer, grid=grid
            )
