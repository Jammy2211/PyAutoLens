import autoarray as aa
import autolens as al
from autolens.data import observation as obs

import numpy as np
import os
import shutil

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


def test__constructor_and_specific_instrument_class_methods(self):

    psf = aa.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=0.1)

    observation = obs.ImagingObservation(
        shape=(51, 51),
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

    lsst = obs.ImagingObservation.lsst()

    lsst_psf = aa.kernel.from_gaussian(shape_2d=(31, 31), sigma=0.5, pixel_scales=0.2)

    assert lsst.shape == (101, 101)
    assert lsst.pixel_scales == (0.2, 0.2)
    assert lsst.psf == lsst_psf
    assert lsst.exposure_time == 100.0
    assert lsst.background_sky_level == 1.0

    euclid = obs.ImagingObservation.euclid()

    euclid_psf = aa.kernel.from_gaussian(shape_2d=(31, 31), sigma=0.1, pixel_scales=0.1)

    assert euclid.shape == (151, 151)
    assert euclid.pixel_scales == (0.1, 0.1)
    assert euclid.psf == euclid_psf
    assert euclid.exposure_time == 565.0
    assert euclid.background_sky_level == 1.0

    hst = obs.ImagingObservation.hst()

    hst_psf = aa.kernel.from_gaussian(shape_2d=(31, 31), sigma=0.05, pixel_scales=0.05)

    assert hst.shape == (251, 251)
    assert hst.pixel_scales == (0.05, 0.05)
    assert hst.psf == hst_psf
    assert hst.exposure_time == 2000.0
    assert hst.background_sky_level == 1.0

    hst_up_sampled = obs.ImagingObservation.hst_up_sampled()

    hst_up_sampled_psf = aa.kernel.from_gaussian(
        shape_2d=(31, 31), sigma=0.05, pixel_scales=0.03
    )

    assert hst_up_sampled.shape == (401, 401)
    assert hst_up_sampled.pixel_scales == (0.03, 0.03)
    assert hst_up_sampled.psf == hst_up_sampled_psf
    assert hst_up_sampled.exposure_time == 2000.0
    assert hst_up_sampled.background_sky_level == 1.0

    adaptive_optics = obs.ImagingObservation.keck_adaptive_optics()

    adaptive_optics_psf = aa.kernel.from_gaussian(
        shape_2d=(31, 31), sigma=0.025, pixel_scales=0.01
    )

    assert adaptive_optics.shape == (751, 751)
    assert adaptive_optics.pixel_scales == (0.01, 0.01)
    assert adaptive_optics.psf == adaptive_optics_psf
    assert adaptive_optics.exposure_time == 1000.0
    assert adaptive_optics.background_sky_level == 1.0

def test__from_deflections_and_galaxies__same_as_manual_calculation_using_tracer(
    self
):

    grid = aa.grid.uniform(
        shape_2d=(10, 10), pixel_scales=1.0, sub_size=1
    )

    g0 = imaging.Galaxy(
        redshift=0.5,
        mass_profile=imaging.mass_profiles.SphericalIsothermal(einstein_radius=1.0),
    )

    g1 = imaging.Galaxy(
        redshift=1.0, light=imaging.light_profiles.SphericalSersic(intensity=1.0)
    )

    tracer = imaging.Tracer.from_galaxies(galaxies=[g0, g1])

    deflections = tracer.deflections_from_grid(grid=grid)

    imaging_data_simulated_via_deflections = imaging.SimulatedImaging.from_deflections_galaxies_and_exposure_arrays(
        deflections=deflections,
        pixel_scales=1.0,
        galaxies=[g1],
        exposure_time=10000.0,
        background_sky_level=100.0,
        add_noise=True,
        noise_seed=1,
    )

    tracer_profile_image = tracer.profile_image_from_grid(grid=grid)

    imaging_data_simulated = imaging.SimulatedImaging.simulate(
        image=tracer_profile_image,
        exposure_time=10000.0,
        background_sky_level=100.0,
        add_noise=True,
        noise_seed=1,
    )

    assert (
        imaging_data_simulated_via_deflections.image == imaging_data_simulated.image
    ).all()
    assert (
        imaging_data_simulated_via_deflections.psf
        == imaging_data_simulated.psf
    ).all()
    assert (
        imaging_data_simulated_via_deflections.noise_map
        == imaging_data_simulated.noise_map
    ).all()
    assert (
        imaging_data_simulated_via_deflections.background_sky_map
        == imaging_data_simulated.background_sky_map
    ).all()
    assert (
        imaging_data_simulated_via_deflections.exposure_time_map
        == imaging_data_simulated.exposure_time_map
    ).all()

def test__from_tracer__same_as_manual_tracer_input(self):
    psf = aa.kernel.manual_2d(
        array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
        pixel_scales=1.0,
    )

    grid = aa.grid.uniform(
        shape_2d=(20, 20), pixel_scales=0.05, sub_size=1
    )

    lens_galaxy = imaging.Galaxy(
        redshift=0.5,
        light=imaging.light_profiles.EllipticalSersic(intensity=1.0),
        mass=imaging.mass_profiles.EllipticalIsothermal(einstein_radius=1.6),
    )

    source_galaxy = imaging.Galaxy(
        redshift=1.0, light=imaging.light_profiles.EllipticalSersic(intensity=0.3)
    )

    tracer = imaging.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    imaging_data_simulated_via_tracer = imaging.SimulatedImaging.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        exposure_time=10000.0,
        psf=psf,
        background_sky_level=100.0,
        add_noise=True,
        noise_seed=1,
    )

    imaging_data_simulated = imaging.SimulatedImaging.simulate(
        image=tracer.padded_profile_image_2d_from_grid_and_psf_shape(
            grid=grid, psf_shape=(3, 3)
        ),
        exposure_time=10000.0,
        psf=psf,
        background_sky_level=100.0,
        add_noise=True,
        noise_seed=1,
    )

    assert (
        imaging_data_simulated_via_tracer.image == imaging_data_simulated.image
    ).all()
    assert (
        imaging_data_simulated_via_tracer.psf == imaging_data_simulated.psf
    ).all()
    assert (
        imaging_data_simulated_via_tracer.noise_map
        == imaging_data_simulated.noise_map
    ).all()
    assert (
        imaging_data_simulated_via_tracer.background_sky_map
        == imaging_data_simulated.background_sky_map
    ).all()
    assert (
        imaging_data_simulated_via_tracer.exposure_time_map
        == imaging_data_simulated.exposure_time_map
    ).all()

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

    grid = aa.grid.uniform(
        shape_2d=(11, 11), pixel_scales=0.2, sub_size=1
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    shape = (11, 11)
    pixel_scales = 0.2
    psf = aa.kernel.from_gaussian(shape_2d=(7, 7), sigma=0.1, pixel_scales=0.2)
    exposure_time = 100.0
    background_sky_level = 1.0

    imaging_data = al.SimulatedImagingData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        exposure_time=exposure_time,
        psf=psf,
        background_sky_level=background_sky_level,
        add_noise=False,
        noise_if_add_noise_false=0.2,
    )

    observation = obs.ImagingObservation(
        shape=shape,
        pixel_scales=pixel_scales,
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
    assert (observation_imaging_data.noise_map.in_2d == 0.2 * np.ones((11, 11))).all()
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

    grid = aa.grid.uniform(
        shape_2d=(11, 11), pixel_scales=0.2, sub_size=1
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    shape = (11, 11)
    pixel_scales = 0.2
    psf = aa.kernel.from_gaussian(shape_2d=(7, 7), sigma=0.1, pixel_scales=0.2)
    exposure_time = 100.0
    background_sky_level = 1.0

    imaging_data = al.SimulatedImagingData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        exposure_time=exposure_time,
        psf=psf,
        background_sky_level=background_sky_level,
        add_noise=False,
        noise_if_add_noise_false=0.2,
    )

    observation = obs.ImagingObservation(
        shape=shape,
        pixel_scales=pixel_scales,
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
        pixel_scales=0.2,
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
    assert (imaging_data.noise_map.in_2d == 0.2 * np.ones((11, 11))).all()
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


def test__uv_from_deflections_and_galaxies__same_as_manual_calculation_using_tracer(
    self, transformer_7x7_7
):

    grid = aa.grid.uniform(
        shape_2d=(10, 10), pixel_scales=1.0, sub_size=1
    )

    g0 = al.Galaxy(
        redshift=0.5,
        mass_profile=al.mass_profiles.SphericalIsothermal(einstein_radius=1.0),
    )

    g1 = al.Galaxy(
        redshift=1.0, light=al.light_profiles.SphericalSersic(intensity=1.0)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[g0, g1])

    deflections = tracer.deflections_from_grid(grid=grid)

    uv_plane_data_simulated_via_deflections = aa.interferometer.from_deflections_galaxies_and_exposure_arrays(
        deflections=deflections,
        pixel_scales=1.0,
        galaxies=[g1],
        exposure_time=10000.0,
        background_sky_level=100.0,
        transformer=transformer_7x7_7,
        noise_sigma=0.1,
        noise_seed=1,
    )

    tracer_profile_image = tracer.profile_image_from_grid(grid=grid)

    uv_plane_data_simulated = aa.interferometer.simulate(
        image=tracer_profile_image,
        pixel_scales=1.0,
        exposure_time=10000.0,
        background_sky_level=100.0,
        transformer=transformer_7x7_7,
        noise_sigma=0.1,
        noise_seed=1,
    )

    assert (
        uv_plane_data_simulated_via_deflections.exposure_time_map
        == uv_plane_data_simulated.exposure_time_map
    ).all()
    assert (
        uv_plane_data_simulated_via_deflections.visibilities
        == uv_plane_data_simulated.visibilities
    ).all()

    assert (
        uv_plane_data_simulated_via_deflections.noise_map
        == uv_plane_data_simulated.noise_map
    ).all()

def test__uv_from_tracer__same_as_manual_tracer_input(self, transformer_7x7_7):

    grid = aa.grid.uniform(
        shape_2d=(20, 20), pixel_scales=0.05, sub_size=1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.light_profiles.EllipticalSersic(intensity=1.0),
        mass=al.mass_profiles.EllipticalIsothermal(einstein_radius=1.6),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.light_profiles.EllipticalSersic(intensity=0.3)
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    uv_plane_data_simulated_via_tracer = aa.interferometer.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scales=0.1,
        exposure_time=10000.0,
        background_sky_level=100.0,
        transformer=transformer_7x7_7,
        noise_sigma=0.1,
        noise_seed=1,
    )

    uv_plane_data_simulated = aa.interferometer.simulate(
        image=tracer.profile_image_from_grid(grid=grid),
        pixel_scales=0.1,
        exposure_time=10000.0,
        background_sky_level=100.0,
        transformer=transformer_7x7_7,
        noise_sigma=0.1,
        noise_seed=1,
    )

    assert (
        uv_plane_data_simulated_via_tracer.exposure_time_map
        == uv_plane_data_simulated.exposure_time_map
    ).all()
    assert (
        uv_plane_data_simulated_via_tracer.visibilities
        == uv_plane_data_simulated.visibilities
    ).all()

    assert (
        uv_plane_data_simulated_via_tracer.noise_map
        == uv_plane_data_simulated.noise_map
    ).all()
