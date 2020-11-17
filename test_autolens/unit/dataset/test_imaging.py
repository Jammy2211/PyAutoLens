import autolens as al
import numpy as np


class TestMaskedImaging:
    def test__masked_dataset_via_autoarray(self, imaging_7x7, sub_mask_7x7):

        masked_imaging_7x7 = al.MaskedImaging(imaging=imaging_7x7, mask=sub_mask_7x7)

        assert (masked_imaging_7x7.image.in_1d == np.ones(9)).all()

        assert (
            masked_imaging_7x7.image.in_2d == np.ones((7, 7)) * np.invert(sub_mask_7x7)
        ).all()

        assert (masked_imaging_7x7.noise_map.in_1d == 2.0 * np.ones(9)).all()
        assert (
            masked_imaging_7x7.noise_map.in_2d
            == 2.0 * np.ones((7, 7)) * np.invert(sub_mask_7x7)
        ).all()

        assert (masked_imaging_7x7.psf.in_1d == (1.0 / 9.0) * np.ones(9)).all()
        assert (masked_imaging_7x7.psf.in_2d == (1.0 / 9.0) * np.ones((3, 3))).all()
        assert masked_imaging_7x7.psf.shape_2d == (3, 3)

        assert type(masked_imaging_7x7.convolver) == al.Convolver

    def test__inheritance_from_autoarray(
        self, imaging_7x7, sub_mask_7x7, blurring_grid_7x7
    ):

        masked_imaging_7x7 = al.MaskedImaging(
            imaging=imaging_7x7,
            mask=sub_mask_7x7,
            settings=al.SettingsMaskedImaging(psf_shape_2d=(3, 3)),
        )

        grid = al.Grid.from_mask(mask=sub_mask_7x7)

        assert (masked_imaging_7x7.grid == grid).all()

        blurring_grid = grid.blurring_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        assert (masked_imaging_7x7.blurring_grid.in_1d == blurring_grid_7x7).all()
        assert (masked_imaging_7x7.blurring_grid == blurring_grid).all()


class TestSimulatorImaging:
    def test__from_tracer_and_grid__same_as_tracer_image(self):
        psf = al.Kernel.from_gaussian(shape_2d=(7, 7), sigma=0.5, pixel_scales=1.0)

        grid = al.Grid.uniform(shape_2d=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(intensity=1.0),
            mass=al.mp.EllipticalIsothermal(einstein_radius=1.6),
        )

        source_galaxy = al.Galaxy(
            redshift=1.0, light=al.lp.EllipticalSersic(intensity=0.3)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.SimulatorImaging(
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_poisson_noise=False,
        )

        imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

        imaging_via_image = simulator.from_image(
            image=tracer.image_from_grid(grid=grid)
        )

        assert imaging.shape_2d == (20, 20)
        assert imaging.image.in_2d[0, 0] != imaging_via_image.image.in_2d[0, 0]
        assert imaging.image.in_2d[10, 10] == imaging_via_image.image.in_2d[10, 10]
        assert (imaging.psf == imaging_via_image.psf).all()
        assert (imaging.noise_map == imaging_via_image.noise_map).all()

    def test__from_deflections_and_galaxies__same_as_calculation_using_tracer(self):

        psf = al.Kernel.no_blur(pixel_scales=0.05)

        grid = al.Grid.uniform(shape_2d=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.Galaxy(
            redshift=0.5, mass=al.mp.EllipticalIsothermal(einstein_radius=1.6)
        )

        source_galaxy = al.Galaxy(
            redshift=1.0, light=al.lp.EllipticalSersic(intensity=0.3)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.SimulatorImaging(
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_poisson_noise=False,
        )

        imaging = simulator.from_deflections_and_galaxies(
            deflections=tracer.deflections_from_grid(grid=grid),
            galaxies=[source_galaxy],
        )

        imaging_via_image = simulator.from_image(
            image=tracer.image_from_grid(grid=grid)
        )

        assert imaging.shape_2d == (20, 20)
        assert (imaging.image.in_2d == imaging_via_image.image.in_2d).all()
        assert (imaging.psf == imaging_via_image.psf).all()
        assert (imaging.noise_map == imaging_via_image.noise_map).all()

    def test__simulate_imaging_from_lens__source_galaxy__compare_to_imaging(self):

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
            ),
        )

        source_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(
                centre=(0.1, 0.1),
                elliptical_comps=(0.096225, -0.055555),
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        grid = al.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.2, sub_size=1)

        psf = al.Kernel.no_blur(pixel_scales=0.2)

        simulator = al.SimulatorImaging(
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_poisson_noise=True,
            noise_seed=1,
        )

        imaging = simulator.from_galaxies_and_grid(
            galaxies=[lens_galaxy, source_galaxy], grid=grid
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        imaging_via_image = simulator.from_image(
            image=tracer.image_from_grid(grid=grid)
        )

        assert imaging.shape_2d == (11, 11)
        assert (imaging.image.in_2d == imaging_via_image.image.in_2d).all()
        assert (imaging.psf == imaging_via_image.psf).all()
        assert (imaging.noise_map == imaging_via_image.noise_map).all()
