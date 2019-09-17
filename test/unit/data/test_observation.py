import autolens as al
from autolens.data import observation as obs

import numpy as np
import os
import shutil

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestObservation:
    def test__constructor_and_specific_instrument_class_methods(self):

        psf = al.PSF.from_gaussian(shape=(11, 11), sigma=0.1, pixel_scale=0.1)

        observation = obs.ImagingObservation(
            shape=(51, 51),
            pixel_scale=0.1,
            psf=psf,
            exposure_time=20.0,
            background_sky_level=10.0,
        )

        assert observation.shape == (51, 51)
        assert observation.pixel_scale == 0.1
        assert observation.psf == psf
        assert observation.exposure_time == 20.0
        assert observation.background_sky_level == 10.0

        lsst = obs.ImagingObservation.lsst()

        lsst_psf = al.PSF.from_gaussian(shape=(31, 31), sigma=0.5, pixel_scale=0.2)

        assert lsst.shape == (101, 101)
        assert lsst.pixel_scale == 0.2
        assert lsst.psf == lsst_psf
        assert lsst.exposure_time == 100.0
        assert lsst.background_sky_level == 1.0

        euclid = obs.ImagingObservation.euclid()

        euclid_psf = al.PSF.from_gaussian(shape=(31, 31), sigma=0.1, pixel_scale=0.1)

        assert euclid.shape == (151, 151)
        assert euclid.pixel_scale == 0.1
        assert euclid.psf == euclid_psf
        assert euclid.exposure_time == 565.0
        assert euclid.background_sky_level == 1.0

        hst = obs.ImagingObservation.hst()

        hst_psf = al.PSF.from_gaussian(shape=(31, 31), sigma=0.05, pixel_scale=0.05)

        assert hst.shape == (251, 251)
        assert hst.pixel_scale == 0.05
        assert hst.psf == hst_psf
        assert hst.exposure_time == 2000.0
        assert hst.background_sky_level == 1.0

        hst_up_sampled = obs.ImagingObservation.hst_up_sampled()

        hst_up_sampled_psf = al.PSF.from_gaussian(
            shape=(31, 31), sigma=0.05, pixel_scale=0.03
        )

        assert hst_up_sampled.shape == (401, 401)
        assert hst_up_sampled.pixel_scale == 0.03
        assert hst_up_sampled.psf == hst_up_sampled_psf
        assert hst_up_sampled.exposure_time == 2000.0
        assert hst_up_sampled.background_sky_level == 1.0

        adaptive_optics = obs.ImagingObservation.keck_adaptive_optics()

        adaptive_optics_psf = al.PSF.from_gaussian(
            shape=(31, 31), sigma=0.025, pixel_scale=0.01
        )

        assert adaptive_optics.shape == (751, 751)
        assert adaptive_optics.pixel_scale == 0.01
        assert adaptive_optics.psf == adaptive_optics_psf
        assert adaptive_optics.exposure_time == 1000.0
        assert adaptive_optics.background_sky_level == 1.0

    def test__simulate_imaging_data_from_lens__source_galaxy__compare_to_manual_imaging_data(
        self
    ):

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mass_profiles.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
            ),
        )

        source_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.light_profiles.EllipticalSersic(
                centre=(0.1, 0.1),
                axis_ratio=0.8,
                phi=60.0,
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(11, 11), pixel_scale=0.2, sub_size=1
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        shape = (11, 11)
        pixel_scale = 0.2
        psf = al.PSF.from_gaussian(shape=(7, 7), sigma=0.1, pixel_scale=0.2)
        exposure_time = 100.0
        background_sky_level = 1.0

        imaging_data = al.SimulatedImagingData.from_tracer_grid_and_exposure_arrays(
            tracer=tracer,
            grid=grid,
            pixel_scale=pixel_scale,
            exposure_time=exposure_time,
            psf=psf,
            background_sky_level=background_sky_level,
            add_noise=False,
            noise_if_add_noise_false=0.2,
        )

        observation = obs.ImagingObservation(
            shape=shape,
            pixel_scale=pixel_scale,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
        )

        observation_imaging_data = observation.simulate_imaging_data_from_galaxies(
            galaxies=[lens_galaxy, source_galaxy],
            sub_size=1,
            add_noise=False,
            noise_if_add_noise_false=0.2,
        )

        assert (imaging_data.image == observation_imaging_data.image).all()
        assert (imaging_data.psf == observation_imaging_data.psf).all()
        assert observation_imaging_data.noise_map == 0.2 * np.ones((11, 11))
        assert imaging_data.noise_map == observation_imaging_data.noise_map
        assert (
            imaging_data.background_noise_map
            == observation_imaging_data.background_noise_map
        )
        assert (
            imaging_data.poisson_noise_map == observation_imaging_data.poisson_noise_map
        )
        assert (
            imaging_data.exposure_time_map == observation_imaging_data.exposure_time_map
        ).all()
        assert (
            imaging_data.background_sky_map
            == observation_imaging_data.background_sky_map
        ).all()

        imaging_data = al.SimulatedImagingData.from_tracer_grid_and_exposure_arrays(
            tracer=tracer,
            grid=grid,
            pixel_scale=pixel_scale,
            exposure_time=exposure_time,
            psf=psf,
            background_sky_level=background_sky_level,
            add_noise=True,
            noise_seed=1,
        )

        observation_imaging_data = observation.simulate_imaging_data_from_galaxies(
            galaxies=[lens_galaxy, source_galaxy],
            sub_size=1,
            add_noise=True,
            noise_seed=1,
        )

        assert (imaging_data.image == observation_imaging_data.image).all()
        assert (imaging_data.psf == observation_imaging_data.psf).all()
        assert (imaging_data.noise_map == observation_imaging_data.noise_map).all()
        assert (
            imaging_data.background_noise_map
            == observation_imaging_data.background_noise_map
        ).all()
        assert (
            imaging_data.poisson_noise_map == observation_imaging_data.poisson_noise_map
        ).all()
        assert (
            imaging_data.exposure_time_map == observation_imaging_data.exposure_time_map
        ).all()
        assert (
            imaging_data.background_sky_map
            == observation_imaging_data.background_sky_map
        ).all()

    def test__simulate_imaging_data_from_lens__source_galaxy__and_write_to_fits(self):

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mass_profiles.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
            ),
        )

        source_galaxy = al.Galaxy(
            redshift=0.5,
            light=al.light_profiles.EllipticalSersic(
                centre=(0.1, 0.1),
                axis_ratio=0.8,
                phi=60.0,
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        grid = al.Grid.from_shape_pixel_scale_and_sub_size(
            shape=(11, 11), pixel_scale=0.2, sub_size=1
        )

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

        shape = (11, 11)
        pixel_scale = 0.2
        psf = al.PSF.from_gaussian(shape=(7, 7), sigma=0.1, pixel_scale=0.2)
        exposure_time = 100.0
        background_sky_level = 1.0

        imaging_data = al.SimulatedImagingData.from_tracer_grid_and_exposure_arrays(
            tracer=tracer,
            grid=grid,
            pixel_scale=pixel_scale,
            exposure_time=exposure_time,
            psf=psf,
            background_sky_level=background_sky_level,
            add_noise=False,
            noise_if_add_noise_false=0.2,
        )

        observation = obs.ImagingObservation(
            shape=shape,
            pixel_scale=pixel_scale,
            psf=psf,
            exposure_time=exposure_time,
            background_sky_level=background_sky_level,
        )

        output_data_dir = "{}/../test_files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        observation.simulate_imaging_data_from_galaxies_and_write_to_fits(
            galaxies=[lens_galaxy, source_galaxy],
            data_path=output_data_dir,
            data_name="observation",
            sub_size=1,
            add_noise=False,
            noise_if_add_noise_false=0.2,
        )

        output_data_dir += "observation/"

        observation_imaging_data_loaded = al.load_imaging_data_from_fits(
            image_path=output_data_dir + "image.fits",
            pixel_scale=0.2,
            psf_path=output_data_dir + "psf.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            background_noise_map_path=output_data_dir + "background_noise_map.fits",
            poisson_noise_map_path=output_data_dir + "poisson_noise_map.fits",
            exposure_time_map_path=output_data_dir + "exposure_time_map.fits",
            background_sky_map_path=output_data_dir + "background_sky_map.fits",
            renormalize_psf=False,
        )

        assert (imaging_data.image == observation_imaging_data_loaded.image).all()
        assert (imaging_data.psf == observation_imaging_data_loaded.psf).all()
        assert imaging_data.noise_map == 0.2 * np.ones((11, 11))
        assert imaging_data.noise_map == observation_imaging_data_loaded.noise_map
        assert (
            imaging_data.background_noise_map
            == observation_imaging_data_loaded.background_noise_map
        )
        assert (
            imaging_data.poisson_noise_map
            == observation_imaging_data_loaded.poisson_noise_map
        )
        assert (
            imaging_data.exposure_time_map
            == observation_imaging_data_loaded.exposure_time_map
        ).all()
        assert (
            imaging_data.background_sky_map
            == observation_imaging_data_loaded.background_sky_map
        ).all()
