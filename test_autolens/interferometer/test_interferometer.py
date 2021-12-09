import autolens as al
import numpy as np
import pytest


class TestSimulatorInterferometer:
    def test__from_tracer__same_as_tracer_input(self):

        grid = al.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllSersic(intensity=1.0),
            mass=al.mp.EllIsothermal(einstein_radius=1.6),
        )

        source_galaxy = al.Galaxy(redshift=1.0, light=al.lp.EllSersic(intensity=0.3))

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.SimulatorInterferometer(
            uv_wavelengths=np.ones(shape=(7, 2)),
            exposure_time=10000.0,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.via_tracer_from(tracer=tracer, grid=grid)

        interferometer_via_image = simulator.via_image_from(
            image=tracer.image_2d_from(grid=grid)
        )

        assert (
            interferometer.visibilities == interferometer_via_image.visibilities
        ).all()
        assert (
            interferometer.uv_wavelengths == interferometer_via_image.uv_wavelengths
        ).all()
        assert (interferometer.noise_map == interferometer_via_image.noise_map).all()

    def test__via_deflections_and_galaxies_from__same_as_calculation_using_tracer(self):

        grid = al.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.Galaxy(
            redshift=0.5, mass=al.mp.EllIsothermal(einstein_radius=1.6)
        )

        source_galaxy = al.Galaxy(redshift=1.0, light=al.lp.EllSersic(intensity=0.3))

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.SimulatorInterferometer(
            uv_wavelengths=np.ones(shape=(7, 2)),
            exposure_time=10000.0,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.via_deflections_and_galaxies_from(
            deflections=tracer.deflections_yx_2d_from(grid=grid),
            galaxies=[source_galaxy],
        )

        interferometer_via_image = simulator.via_image_from(
            image=tracer.image_2d_from(grid=grid)
        )

        assert (
            interferometer.visibilities == interferometer_via_image.visibilities
        ).all()
        assert (
            interferometer_via_image.uv_wavelengths == interferometer.uv_wavelengths
        ).all()
        assert (interferometer.noise_map == interferometer_via_image.noise_map).all()

    def test__simulate_interferometer_from_lens__source_galaxy__compare_to_interferometer(
        self,
    ):

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
            ),
        )

        source_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllSersic(
                centre=(0.1, 0.1),
                elliptical_comps=(0.096225, -0.055555),
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.05, sub_size=1)

        simulator = al.SimulatorInterferometer(
            uv_wavelengths=np.ones(shape=(7, 2)),
            exposure_time=10000.0,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.via_galaxies_from(
            galaxies=[lens_galaxy, source_galaxy], grid=grid
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        interferometer_via_image = simulator.via_image_from(
            image=tracer.image_2d_from(grid=grid)
        )

        assert interferometer.visibilities == pytest.approx(
            interferometer_via_image.visibilities, 1.0e-4
        )
        assert (
            interferometer.uv_wavelengths == interferometer_via_image.uv_wavelengths
        ).all()
        assert (interferometer_via_image.noise_map == interferometer.noise_map).all()
