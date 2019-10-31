import autolens as al

import numpy as np
import os
import shutil

test_data_dir = "{}/../test_files/arrays/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestSimulatorImaging:

    def test__from_tracer__same_as_manual_tracer_input(self):
        psf = al.kernel.manual_2d(
            array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
            pixel_scales=1.0,
        )

        grid = al.grid.uniform(shape_2d=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(intensity=1.0),
            mass=al.mp.EllipticalIsothermal(einstein_radius=1.6),
        )

        source_galaxy = al.galaxy(
            redshift=1.0, light=al.lp.EllipticalSersic(intensity=0.3)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.ImagingSimulator(
            shape_2d=(20, 20),
            pixel_scales=0.05,
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_noise=True, 
            noise_seed=1
        )

        imaging_simulated = simulator.from_tracer_and_grid(
            tracer=tracer, grid=grid,
        )

        imaging_manual = al.imaging.simulate(
            image=tracer.padded_profile_image_from_grid_and_psf_shape(
                grid=grid, psf_shape=(3, 3)
            ),
            exposure_time=10000.0,
            psf=psf,
            background_sky_level=100.0,
            add_noise=True,
            noise_seed=1,
        )

        assert (imaging_simulated.image.in_2d == imaging_manual.image.in_2d).all()
        assert (imaging_simulated.psf == imaging_manual.psf).all()
        assert (imaging_simulated.noise_map == imaging_manual.noise_map).all()
        assert (
            imaging_simulated.background_sky_map == imaging_manual.background_sky_map
        ).all()
        assert (
            imaging_simulated.exposure_time_map == imaging_manual.exposure_time_map
        ).all()

    def test__from_deflections_and_galaxies__same_as_manual_calculation_using_tracer(
        self
    ):

        psf = al.kernel.manual_2d(
            array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
            pixel_scales=0.05,
        )

        grid = al.grid.uniform(shape_2d=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.galaxy(
            redshift=0.5, mass=al.mp.EllipticalIsothermal(einstein_radius=1.6)
        )

        source_galaxy = al.galaxy(
            redshift=1.0, light=al.lp.EllipticalSersic(intensity=0.3)
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        simulator = al.ImagingSimulator(
            shape_2d=(20, 20),
            pixel_scales=0.05,
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_noise=True,
            noise_seed=1,
        )

        imaging_simulated = simulator.from_deflections_and_galaxies(
            deflections=tracer.deflections_from_grid(grid=grid),
            galaxies=[source_galaxy],
        )

        imaging_manual = al.imaging.simulate(
            image=tracer.padded_profile_image_from_grid_and_psf_shape(
                grid=grid, psf_shape=(1, 1)
            ),
            exposure_time=10000.0,
            psf=psf,
            background_sky_level=100.0,
            add_noise=True,
            noise_seed=1,
        )

        assert (imaging_simulated.image.in_2d == imaging_manual.image.in_2d).all()
        assert (imaging_simulated.psf == imaging_manual.psf).all()
        assert (imaging_simulated.noise_map == imaging_manual.noise_map).all()
        assert (
            imaging_simulated.background_sky_map == imaging_manual.background_sky_map
        ).all()
        assert (
            imaging_simulated.exposure_time_map == imaging_manual.exposure_time_map
        ).all()

    def test__simulate_imaging_from_lens__source_galaxy__compare_to_manual_imaging(
        self
    ):

        lens_galaxy = al.galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
            ),
        )

        source_galaxy = al.galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(
                centre=(0.1, 0.1),
                axis_ratio=0.8,
                phi=60.0,
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        grid = al.grid.uniform(shape_2d=(11, 11), pixel_scales=0.2, sub_size=1)

        psf = al.kernel.from_gaussian(shape_2d=(7, 7), sigma=0.1, pixel_scales=0.2)

        simulator = al.ImagingSimulator(
            shape_2d=(11, 11),
            pixel_scales=0.2,
            psf=psf,
            exposure_time=100.0,
            background_sky_level=1.0,
            add_noise=False,
            noise_if_add_noise_false=0.2,
        )

        imaging_simulated = simulator.from_galaxies(
            galaxies=[lens_galaxy, source_galaxy],
            sub_size=1,
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        imaging_manual = al.imaging.simulate(
            image=tracer.padded_profile_image_from_grid_and_psf_shape(
                grid=grid, psf_shape=(7, 7)
            ),
            exposure_time=100.0,
            psf=psf,
            background_sky_level=1.0,
            add_noise=False,
            noise_if_add_noise_false=0.2,
        )

        assert (imaging_manual.image == imaging_simulated.image).all()
        assert (imaging_manual.psf == imaging_simulated.psf).all()
        assert (imaging_simulated.noise_map.in_2d == 0.2 * np.ones((11, 11))).all()
        assert imaging_manual.noise_map == imaging_simulated.noise_map
        assert (
            imaging_manual.background_noise_map
            == imaging_simulated.background_noise_map
        )
        assert imaging_manual.poisson_noise_map == imaging_simulated.poisson_noise_map
        assert (
            imaging_manual.exposure_time_map == imaging_simulated.exposure_time_map
        ).all()
        assert (
            imaging_manual.background_sky_map == imaging_simulated.background_sky_map
        ).all()

    def test__simulate_imaging_from_lens__source_galaxy__and_write_to_fits(self):

        lens_galaxy = al.galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
            ),
        )

        source_galaxy = al.galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(
                centre=(0.1, 0.1),
                axis_ratio=0.8,
                phi=60.0,
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        psf = al.kernel.from_gaussian(shape_2d=(7, 7), sigma=0.1, pixel_scales=0.2)

        simulator = al.ImagingSimulator(
            shape_2d=(11, 11),
            pixel_scales=0.2,
            psf=psf,
            exposure_time=100.0,
            background_sky_level=1.0,
            add_noise=False,
            noise_if_add_noise_false=0.2,
        )

        imaging_simulated = simulator.from_galaxies(
            galaxies=[lens_galaxy, source_galaxy],
            sub_size=1,
        )

        output_data_dir = "{}/../test_files/arrays/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        simulator.from_galaxies_and_write_to_fits(
            galaxies=[lens_galaxy, source_galaxy],
            data_path=output_data_dir,
            data_name="observation",
            sub_size=1,
        )

        output_data_dir += "observation/"

        simulator_imaging_loaded = al.imaging.from_fits(
            image_path=output_data_dir + "image.fits",
            pixel_scales=0.2,
            psf_path=output_data_dir + "psf.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            background_noise_map_path=output_data_dir + "background_noise_map.fits",
            poisson_noise_map_path=output_data_dir + "poisson_noise_map.fits",
            exposure_time_map_path=output_data_dir + "exposure_time_map.fits",
            background_sky_map_path=output_data_dir + "background_sky_map.fits",
            renormalize_psf=False,
        )

        assert (imaging_simulated.image == simulator_imaging_loaded.image).all()
        assert (imaging_simulated.psf == simulator_imaging_loaded.psf).all()
        assert (imaging_simulated.noise_map.in_2d == 0.2 * np.ones((11, 11))).all()
        assert (
            imaging_simulated.background_noise_map
            == simulator_imaging_loaded.background_noise_map
        )
        assert (
            imaging_simulated.poisson_noise_map
            == simulator_imaging_loaded.poisson_noise_map
        )
        assert (
            imaging_simulated.exposure_time_map
            == simulator_imaging_loaded.exposure_time_map
        ).all()
        assert (
            imaging_simulated.background_sky_map
            == simulator_imaging_loaded.background_sky_map
        ).all()
