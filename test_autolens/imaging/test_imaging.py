import autolens as al
import numpy as np


class TestSimulatorImaging:
    def test__via_tracer_from__same_as_tracer_image(self):
        psf = al.Kernel2D.from_gaussian(
            shape_native=(7, 7), sigma=0.5, pixel_scales=1.0
        )

        grid = al.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllSersic(intensity=1.0),
            mass=al.mp.EllIsothermal(einstein_radius=1.6),
        )

        source_galaxy = al.Galaxy(redshift=1.0, light=al.lp.EllSersic(intensity=0.3))

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.SimulatorImaging(
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_poisson_noise=False,
        )

        imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

        imaging_via_image = simulator.via_image_from(
            image=tracer.image_2d_from(grid=grid)
        )

        assert imaging.shape_native == (20, 20)
        assert imaging.image.native[0, 0] != imaging_via_image.image.native[0, 0]
        assert imaging.image.native[10, 10] == imaging_via_image.image.native[10, 10]
        assert (imaging.psf == imaging_via_image.psf).all()
        assert (imaging.noise_map == imaging_via_image.noise_map).all()

    def test__via_deflections_and_galaxies_from__same_as_calculation_using_tracer(self):

        psf = al.Kernel2D.no_blur(pixel_scales=0.05)

        grid = al.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.Galaxy(
            redshift=0.5, mass=al.mp.EllIsothermal(einstein_radius=1.6)
        )

        source_galaxy = al.Galaxy(redshift=1.0, light=al.lp.EllSersic(intensity=0.3))

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.SimulatorImaging(
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_poisson_noise=False,
        )

        imaging = simulator.via_deflections_and_galaxies_from(
            deflections=tracer.deflections_yx_2d_from(grid=grid),
            galaxies=[source_galaxy],
        )

        imaging_via_image = simulator.via_image_from(
            image=tracer.image_2d_from(grid=grid)
        )

        assert imaging.shape_native == (20, 20)
        assert (imaging.image.native == imaging_via_image.image.native).all()
        assert (imaging.psf == imaging_via_image.psf).all()
        assert (imaging.noise_map == imaging_via_image.noise_map).all()

    def test__simulate_imaging_from_lens__source_galaxy__compare_to_imaging(self):

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

        grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, sub_size=1)

        psf = al.Kernel2D.manual_native(array=[[1.0]], pixel_scales=0.2)

        simulator = al.SimulatorImaging(
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_poisson_noise=True,
            noise_seed=1,
        )

        imaging = simulator.via_galaxies_from(
            galaxies=[lens_galaxy, source_galaxy], grid=grid
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        imaging_via_image = simulator.via_image_from(
            image=tracer.image_2d_from(grid=grid)
        )

        assert imaging.shape_native == (11, 11)
        assert (imaging.image.native == imaging_via_image.image.native).all()
        assert (imaging.psf == imaging_via_image.psf).all()
        assert (imaging.noise_map == imaging_via_image.noise_map).all()
