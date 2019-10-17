import autolens as al
import os
import shutil

import numpy as np
import pytest

import autoarray as aa
from autolens import exc

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestUVPlaneDataFrom(object):
    def test__data_with_resized_primary_beam(self):

        uv_plane_data = al.UVPlaneData(
            shape_2d=(2, 2),
            pixel_scales=1.0,
            visibilities=np.array([[1, 1]]),
            primary_beam=aa.kernel.zeros(shape_2d=(5,5), pixel_scales=1.0),
            noise_map=1,
            exposure_time_map=1,
            uv_wavelengths=1,
        )

        uv_plane_data = uv_plane_data.resized_primary_beam_from_new_shape(
            new_shape=(1, 1)
        )

        assert (uv_plane_data.primary_beam.in_2d == np.zeros((1, 1))).all()

    def test__data_with_modified_visibilities(self):

        uv_plane_data = al.UVPlaneData(
            shape_2d=(2, 2),
            pixel_scales=1.0,
            visibilities=np.array([[1, 1]]),
            primary_beam=aa.kernel.zeros(shape_2d=(5,5), pixel_scales=1.0),
            noise_map=1,
            exposure_time_map=2,
            uv_wavelengths=3,
        )

        uv_plane_data = uv_plane_data.modified_visibilities_from_visibilities(
            visibilities=np.array([[2, 2]])
        )

        assert (uv_plane_data.visibilities == np.array([[2, 2]])).all()
        assert uv_plane_data.shape == (2, 2)
        assert uv_plane_data.pixel_scales == (1.0, 1.0)
        assert (uv_plane_data.primary_beam.in_2d == np.zeros((1, 1))).all()
        assert uv_plane_data.noise_map == 1
        assert uv_plane_data.exposure_time_map == 2
        assert uv_plane_data.uv_wavelengths == 3

    def test__new_data_in_counts__all_arrays_in_units_of_flux_are_converted(self):

        uv_plane_data = al.UVPlaneData(
            shape_2d=(2, 2),
            visibilities=np.ones((3, 2)),
            pixel_scales=1.0,
            noise_map=2.0 * np.ones((3,)),
            exposure_time_map=0.5 * np.ones((3,)),
            primary_beam=1,
            uv_wavelengths=1,
        )

        uv_plane_data = uv_plane_data.data_in_electrons()

        assert (uv_plane_data.visibilities == 2.0 * np.ones((3, 2))).all()
        assert (uv_plane_data.noise_map == 4.0 * np.ones((3,))).all()

    def test__new_data_in_adus__all_arrays_in_units_of_flux_are_converted(self):

        uv_plane_data = al.UVPlaneData(
            shape_2d=(2, 2),
            visibilities=np.ones((3, 2)),
            pixel_scales=1.0,
            noise_map=2.0 * np.ones((3,)),
            exposure_time_map=0.5 * np.ones((3)),
            primary_beam=1,
            uv_wavelengths=1,
        )

        uv_plane_data = uv_plane_data.data_in_adus_from_gain(gain=2.0)

        assert (uv_plane_data.visibilities == 2.0 * 2.0 * np.ones((3, 2))).all()
        assert (uv_plane_data.noise_map == 2.0 * 4.0 * np.ones((3,))).all()


class TestSimulateUVPlaneData(object):
    def test__setup_with_all_features_off(self, transformer_7x7_7):
        image = aa.array.manual_2d([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])

        exposure_time_map = aa.array.full(
            fill_value=1.0, pixel_scales=0.1, shape_2d=image.shape_2d
        )

        uv_plane_data_simulated = al.SimulatedUVPlaneData.from_image_and_exposure_arrays(
            image=image,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            pixel_scales=0.1,
            transformer=transformer_7x7_7,
            noise_sigma=None,
        )

        simulated_visibilities = transformer_7x7_7.visibilities_from_image(
            image=image
        )

        assert uv_plane_data_simulated.visibilities == pytest.approx(
            simulated_visibilities, 1.0e-4
        )
        assert uv_plane_data_simulated.pixel_scales == (0.1, 0.1)

    def test__setup_with_background_sky_on__noise_off__no_noise_in_image__noise_map_is_noise_value(
        self, transformer_7x7_7
    ):
        image = aa.array.manual_2d([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])

        exposure_time_map = aa.array.full(
            fill_value=1.0, pixel_scales=0.1, shape_2d=image.shape_2d
        )

        background_sky_map = aa.array.full(
            fill_value=2.0, pixel_scales=0.1, shape_2d=image.shape_2d
        )

        uv_plane_data_simulated = al.SimulatedUVPlaneData.from_image_and_exposure_arrays(
            image=image,
            pixel_scales=0.1,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            transformer=transformer_7x7_7,
            noise_sigma=None,
            noise_if_add_noise_false=0.2,
            noise_seed=1,
        )

        simulated_visibilities = transformer_7x7_7.visibilities_from_image(
            image=image + background_sky_map
        )

        assert uv_plane_data_simulated.visibilities == pytest.approx(
            simulated_visibilities, 1.0e-4
        )
        assert (
            uv_plane_data_simulated.exposure_time_map.in_2d == 1.0 * np.ones((3, 3))
        ).all()

        assert (uv_plane_data_simulated.noise_map == 0.2 * np.ones((7, 2))).all()
        assert uv_plane_data_simulated.pixel_scales == (0.1, 0.1)

    def test__setup_with_noise(self, transformer_7x7_7):

        image = aa.array.manual_2d([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])

        exposure_time_map = aa.array.full(
            fill_value=20.0, pixel_scales=0.1, shape_2d=image.shape_2d
        )

        uv_plane_data_simulated = al.SimulatedUVPlaneData.from_image_and_exposure_arrays(
            image=image,
            pixel_scales=0.1,
            exposure_time=20.0,
            exposure_time_map=exposure_time_map,
            transformer=transformer_7x7_7,
            noise_sigma=0.1,
            noise_seed=1,
        )

        simulated_visibilities = transformer_7x7_7.visibilities_from_image(
            image=image
        )

        assert (
            uv_plane_data_simulated.exposure_time_map.in_2d == 20.0 * np.ones((3, 3))
        ).all()
        assert uv_plane_data_simulated.pixel_scales == (0.1, 0.1)

        assert uv_plane_data_simulated.visibilities[0, :] == pytest.approx(
            [1.728611, -2.582958], 1.0e-4
        )
        visibilities_noise_map_realization = (
            uv_plane_data_simulated.visibilities - simulated_visibilities
        )

        assert visibilities_noise_map_realization == pytest.approx(
            uv_plane_data_simulated.noise_map_realization, 1.0e-4
        )

        assert (uv_plane_data_simulated.noise_map == 0.1 * np.ones((7, 2))).all()

    def test__from_deflections_and_galaxies__same_as_manual_calculation_using_tracer(
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

        uv_plane_data_simulated_via_deflections = al.SimulatedUVPlaneData.from_deflections_galaxies_and_exposure_arrays(
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

        uv_plane_data_simulated = al.SimulatedUVPlaneData.from_image_and_exposure_arrays(
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

    def test__from_tracer__same_as_manual_tracer_input(self, transformer_7x7_7):

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

        uv_plane_data_simulated_via_tracer = al.SimulatedUVPlaneData.from_tracer_grid_and_exposure_arrays(
            tracer=tracer,
            grid=grid,
            pixel_scales=0.1,
            exposure_time=10000.0,
            background_sky_level=100.0,
            transformer=transformer_7x7_7,
            noise_sigma=0.1,
            noise_seed=1,
        )

        uv_plane_data_simulated = al.SimulatedUVPlaneData.from_image_and_exposure_arrays(
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

    class TestCreateGaussianNoiseMap(object):
        def test__gaussian_noise_sigma_0__gaussian_noise_map_all_0__image_is_identical_to_input(
            self
        ):
            simulate_gaussian_noise = al.gaussian_noise_map_from_shape_and_sigma(
                shape=(9,), sigma=0.0, noise_seed=1
            )

            assert (simulate_gaussian_noise == np.zeros((9,))).all()

        def test__gaussian_noise_sigma_1__gaussian_noise_map_all_non_0__image_has_noise_added(
            self
        ):
            simulate_gaussian_noise = al.gaussian_noise_map_from_shape_and_sigma(
                shape=(9,), sigma=1.0, noise_seed=1
            )

            # Use seed to give us a known gaussian noises map we'll test_autoarray for

            assert simulate_gaussian_noise == pytest.approx(
                np.array([1.62, -0.61, -0.53, -1.07, 0.87, -2.30, 1.74, -0.76, 0.32]),
                1e-2,
            )


class TestUVPlaneFromFits(object):
    def test__no_settings_just_pass_fits(self):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(7, 7),
            pixel_scales=0.1,
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            noise_map_path=test_data_dir + "3_threes.fits",
            u_wavelengths_path=test_data_dir + "3_fours.fits",
            v_wavelengths_path=test_data_dir + "3_fives.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            renormalize_primary_beam=False,
        )

        assert (uv_plane_data.visibilities[:, 0] == np.ones(3)).all()
        assert (uv_plane_data.visibilities[:, 1] == 2.0 * np.ones(3)).all()
        assert (uv_plane_data.noise_map == 3.0 * np.ones(3)).all()
        assert (uv_plane_data.uv_wavelengths[:, 0] == 4.0 * np.ones(3)).all()
        assert (uv_plane_data.uv_wavelengths[:, 1] == 5.0 * np.ones(3)).all()
        assert (uv_plane_data.primary_beam.in_2d == 5.0 * np.ones((3, 3))).all()

        assert uv_plane_data.pixel_scales == (0.1, 0.1)

    def test__optional_array_paths_included__loads_optional_array(self):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(7, 7),
            pixel_scales=0.1,
            noise_map_path=test_data_dir + "3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            renormalize_primary_beam=False,
        )

        assert (uv_plane_data.noise_map == 3.0 * np.ones((3,))).all()
        assert (uv_plane_data.primary_beam.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (uv_plane_data.exposure_time_map == 6.0 * np.ones((3,))).all()

        assert uv_plane_data.pixel_scales == (0.1, 0.1)

    def test__all_files_in_one_fits__load_using_different_hdus(self):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(7, 7),
            pixel_scales=0.1,
            noise_map_path=test_data_dir + "3_multiple_hdu.fits",
            noise_map_hdu=2,
            primary_beam_path=test_data_dir + "3x3_multiple_hdu.fits",
            primary_beam_hdu=3,
            exposure_time_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            exposure_time_map_hdu=5,
            real_visibilities_path=test_data_dir + "3_multiple_hdu.fits",
            real_visibilities_hdu=0,
            imaginary_visibilities_path=test_data_dir + "3_multiple_hdu.fits",
            imaginary_visibilities_hdu=1,
            u_wavelengths_path=test_data_dir + "3_multiple_hdu.fits",
            u_wavelengths_hdu=3,
            v_wavelengths_path=test_data_dir + "3_multiple_hdu.fits",
            v_wavelengths_hdu=4,
            renormalize_primary_beam=False,
        )

        assert (uv_plane_data.primary_beam.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (uv_plane_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()
        assert (uv_plane_data.visibilities[:, 0] == np.ones(3)).all()
        assert (uv_plane_data.visibilities[:, 1] == 2.0 * np.ones(3)).all()
        assert (uv_plane_data.noise_map == 3.0 * np.ones(3)).all()
        assert (uv_plane_data.uv_wavelengths[:, 0] == 4.0 * np.ones(3)).all()
        assert (uv_plane_data.uv_wavelengths[:, 1] == 5.0 * np.ones(3)).all()

        assert uv_plane_data.pixel_scales == (0.1, 0.1)

    def test__exposure_time_included__creates_exposure_time_map_using_exposure_time(
        self
    ):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(7, 7),
            noise_map_path=test_data_dir + "3_ones.fits",
            primary_beam_path=test_data_dir + "3x3_ones.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            pixel_scales=0.1,
            exposure_time_map_from_single_value=3.0,
        )

        assert (uv_plane_data.exposure_time_map == 3.0 * np.ones((3,))).all()

    def test__pad_shape_of_primary_beam(self):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(7, 7),
            pixel_scales=0.1,
            noise_map_path=test_data_dir + "3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            resized_primary_beam_shape=(9, 9),
            renormalize_primary_beam=False,
        )

        primary_beam_padded_array = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        assert (uv_plane_data.primary_beam.in_2d == primary_beam_padded_array).all()

        assert uv_plane_data.pixel_scales == (0.1, 0.1)

    def test__trim_shape_of_primary_beam(self):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(7, 7),
            pixel_scales=0.1,
            noise_map_path=test_data_dir + "3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            resized_primary_beam_shape=(1, 1),
            renormalize_primary_beam=False,
        )

        trimmed_array = np.array([[1.0]])

        assert (uv_plane_data.primary_beam.in_2d == 5.0 * trimmed_array).all()

        assert uv_plane_data.pixel_scales == (0.1, 0.1)

    def test__primary_beam_renormalized_false__does_not_renormalize_primary_beam(self):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(7, 7),
            pixel_scales=0.1,
            noise_map_path=test_data_dir + "3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            renormalize_primary_beam=False,
        )

        assert (uv_plane_data.primary_beam.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (uv_plane_data.noise_map == 3.0 * np.ones((3,))).all()
        assert (uv_plane_data.exposure_time_map == 6.0 * np.ones((3,))).all()

        assert uv_plane_data.pixel_scales == (0.1, 0.1)

    def test__primary_beam_renormalized_true__renormalized_primary_beam(self):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(7, 7),
            pixel_scales=0.1,
            noise_map_path=test_data_dir + "3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            renormalize_primary_beam=True,
        )

        assert uv_plane_data.primary_beam.in_2d == pytest.approx(
            (1.0 / 9.0) * np.ones((3, 3)), 1e-2
        )
        assert (uv_plane_data.noise_map == 3.0 * np.ones((3,))).all()
        assert (uv_plane_data.exposure_time_map == 6.0 * np.ones((3,))).all()

        assert uv_plane_data.pixel_scales == (0.1, 0.1)

    def test__convert_visibilities_from_electrons_using_exposure_time(self):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(2, 2),
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            pixel_scales=0.1,
            noise_map_path=test_data_dir + "3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            renormalize_primary_beam=False,
            convert_from_electrons=True,
        )

        assert (uv_plane_data.visibilities[:, 0] == np.ones((3,)) / 6.0).all()
        assert (uv_plane_data.visibilities[:, 1] == 2.0 * np.ones((3,)) / 6.0).all()
        assert (uv_plane_data.noise_map == 3.0 * np.ones((3,)) / 6.0).all()
        assert (uv_plane_data.exposure_time_map == 6.0 * np.ones((3,))).all()

        assert uv_plane_data.pixel_scales == (0.1, 0.1)

    def test__convert_image_from_adus_using_exposure_time_and_gain(self):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(2, 2),
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            pixel_scales=0.1,
            noise_map_path=test_data_dir + "3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            gain=2.0,
            convert_from_adus=True,
        )

        assert (uv_plane_data.visibilities[:, 0] == 2.0 * np.ones((3,)) / 6.0).all()
        assert (
            uv_plane_data.visibilities[:, 1] == 2.0 * 2.0 * np.ones((3,)) / 6.0
        ).all()
        assert (uv_plane_data.noise_map == 2.0 * 3.0 * np.ones((3, 3)) / 6.0).all()
        assert (uv_plane_data.exposure_time_map == 6.0 * np.ones((3, 3))).all()

        assert uv_plane_data.pixel_scales == (0.1, 0.1)

    def test__exposure_time_and_exposure_time_map_included__raies_imaging_error(self):

        with pytest.raises(exc.DataException):
            al.load_uv_plane_data_from_fits(
                shape=(7, 7),
                real_visibilities_path=test_data_dir + "3_ones.fits",
                imaginary_visibilities_path=test_data_dir + "3_twos.fits",
                pixel_scales=0.1,
                noise_map_path=test_data_dir + "3x3_threes.fits",
                exposure_time_map_path=test_data_dir + "3x3_ones.fits",
                exposure_time_map_from_single_value=1.0,
            )

    def test__output_all_arrays(self):

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(7, 7),
            pixel_scales=0.1,
            real_visibilities_path=test_data_dir + "3_ones.fits",
            imaginary_visibilities_path=test_data_dir + "3_twos.fits",
            noise_map_path=test_data_dir + "3_threes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            u_wavelengths_path=test_data_dir + "3_fours.fits",
            v_wavelengths_path=test_data_dir + "3_fives.fits",
            renormalize_primary_beam=False,
        )

        output_data_dir = "{}/../test_files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        al.output_uv_plane_data_to_fits(
            uv_plane_data=uv_plane_data,
            real_visibilities_path=output_data_dir + "real_visibilities.fits",
            imaginary_visibilities_path=output_data_dir + "imaginary_visibilities.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            primary_beam_path=output_data_dir + "primary_beam.fits",
            exposure_time_map_path=output_data_dir + "exposure_time_map.fits",
            u_wavelengths_path=output_data_dir + "u_wavelengths.fits",
            v_wavelengths_path=output_data_dir + "v_wavelengths.fits",
            overwrite=True,
        )

        uv_plane_data = al.load_uv_plane_data_from_fits(
            shape=(7, 7),
            pixel_scales=0.1,
            real_visibilities_path=output_data_dir + "real_visibilities.fits",
            imaginary_visibilities_path=output_data_dir + "imaginary_visibilities.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            primary_beam_path=output_data_dir + "primary_beam.fits",
            exposure_time_map_path=output_data_dir + "exposure_time_map.fits",
            u_wavelengths_path=output_data_dir + "u_wavelengths.fits",
            v_wavelengths_path=output_data_dir + "v_wavelengths.fits",
            renormalize_primary_beam=False,
        )

        assert (uv_plane_data.noise_map == 3.0 * np.ones((3, 3))).all()
        assert (uv_plane_data.primary_beam.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (uv_plane_data.exposure_time_map == 6.0 * np.ones((3,))).all()
        assert (uv_plane_data.visibilities[:, 0] == np.ones(3)).all()
        assert (uv_plane_data.visibilities[:, 1] == 2.0 * np.ones(3)).all()
        assert (uv_plane_data.noise_map == 3.0 * np.ones(3)).all()
        assert (uv_plane_data.uv_wavelengths[:, 0] == 4.0 * np.ones(3)).all()
        assert (uv_plane_data.uv_wavelengths[:, 1] == 5.0 * np.ones(3)).all()

        assert uv_plane_data.pixel_scales == (0.1, 0.1)
