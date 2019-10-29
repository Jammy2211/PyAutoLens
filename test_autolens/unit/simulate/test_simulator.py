import autolens as al
from autolens.simulate import simulator as obs

import numpy as np
import os
import shutil

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestImaging:
    def test__constructor_and_specific_instrument_class_methods(self):

        psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=0.1)

        observation = obs.ImagingSimulator(
            shape_2d=(51, 51),
            pixel_scales=0.1,
            psf=psf,
            exposure_time=20.0,
            background_sky_level=10.0,
        )

        assert observation.shape == (51, 51)
        assert observation.pixel_scales == (0.1, 0.1)
        assert observation.psf == psf
        assert observation.exposure_time == 20.0
        assert observation.background_sky_level == 10.0

        lsst = obs.ImagingSimulator.lsst()

        lsst_psf = al.kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.5, pixel_scales=0.2
        )

        assert lsst.shape == (101, 101)
        assert lsst.pixel_scales == (0.2, 0.2)
        assert lsst.psf == lsst_psf
        assert lsst.exposure_time == 100.0
        assert lsst.background_sky_level == 1.0

        euclid = obs.ImagingSimulator.euclid()

        euclid_psf = al.kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.1, pixel_scales=0.1
        )

        assert euclid.shape == (151, 151)
        assert euclid.pixel_scales == (0.1, 0.1)
        assert euclid.psf == euclid_psf
        assert euclid.exposure_time == 565.0
        assert euclid.background_sky_level == 1.0

        hst = obs.ImagingSimulator.hst()

        hst_psf = al.kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.05, pixel_scales=0.05
        )

        assert hst.shape == (251, 251)
        assert hst.pixel_scales == (0.05, 0.05)
        assert hst.psf == hst_psf
        assert hst.exposure_time == 2000.0
        assert hst.background_sky_level == 1.0

        hst_up_sampled = obs.ImagingSimulator.hst_up_sampled()

        hst_up_sampled_psf = al.kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.05, pixel_scales=0.03
        )

        assert hst_up_sampled.shape == (401, 401)
        assert hst_up_sampled.pixel_scales == (0.03, 0.03)
        assert hst_up_sampled.psf == hst_up_sampled_psf
        assert hst_up_sampled.exposure_time == 2000.0
        assert hst_up_sampled.background_sky_level == 1.0

        adaptive_optics = obs.ImagingSimulator.keck_adaptive_optics()

        adaptive_optics_psf = al.kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.025, pixel_scales=0.01
        )

        assert adaptive_optics.shape == (751, 751)
        assert adaptive_optics.pixel_scales == (0.01, 0.01)
        assert adaptive_optics.psf == adaptive_optics_psf
        assert adaptive_optics.exposure_time == 1000.0
        assert adaptive_optics.background_sky_level == 1.0

    def test__from_tracer__same_as_manual_tracer_input(self):
        psf = al.kernel.manual_2d(
            array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
            pixel_scales=1.0,
        )

        grid = al.grid.uniform(shape_2d=(20, 20), pixel_scales=0.05, sub_size=1)

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.lp.EllipticalSersic(intensity=1.0),
            mass=al.mp.EllipticalIsothermal(einstein_radius=1.6),
        )

        source_galaxy = al.Galaxy(
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

        lens_galaxy = al.Galaxy(
            redshift=0.5, mass=al.mp.EllipticalIsothermal(einstein_radius=1.6)
        )

        source_galaxy = al.Galaxy(
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

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
            ),
        )

        source_galaxy = al.Galaxy(
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

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
            ),
        )

        source_galaxy = al.Galaxy(
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

        output_data_dir = "{}/../test_files/array/output_test/".format(
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
