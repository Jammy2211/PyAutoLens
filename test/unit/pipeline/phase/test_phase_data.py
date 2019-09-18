import os
from os import path

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit as af
import autolens as al
from autolens import exc
from test.unit.mock.pipeline import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/config/phase_imaging_7x7".format(directory)
    )


def clean_images():
    try:
        os.remove("{}/source_lens_phase/source_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/lens_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/model_image_0.fits".format(directory))
    except FileNotFoundError:
        pass
    af.conf.instance.data_path = directory


class TestPhase(object):
    def test__make_analysis(
        self, phase_imaging_7x7, imaging_data_7x7, lens_imaging_data_7x7
    ):
        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)

        assert analysis.last_results is None
        assert (
            analysis.lens_imaging_data.image(return_in_2d=True, return_masked=False)
            == imaging_data_7x7.image
        )
        assert (
            analysis.lens_imaging_data.noise_map(return_in_2d=True, return_masked=False)
            == imaging_data_7x7.noise_map
        )

    def test__make_analysis__mask_input_uses_mask__no_mask_uses_mask_function(
        self, phase_imaging_7x7, imaging_data_7x7
    ):
        # If an input mask is supplied and there is no mask function, we use mask input.

        phase_imaging_7x7.mask_function = None

        mask_input = al.Mask.circular(
            shape=imaging_data_7x7.shape, pixel_scale=1, sub_size=1, radius_arcsec=1.5
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, mask=mask_input
        )

        assert (analysis.lens_imaging_data.mask == mask_input).all()

        # If a mask function is suppled, we should use this mask, regardless of whether an input mask is supplied.

        def mask_function(image, sub_size):
            return al.Mask.circular(
                shape=image.shape, pixel_scale=1, sub_size=sub_size, radius_arcsec=0.3
            )

        mask_from_function = mask_function(image=imaging_data_7x7.image, sub_size=1)
        phase_imaging_7x7.mask_function = mask_function

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7, mask=None)
        assert (analysis.lens_imaging_data.mask == mask_from_function).all()
        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, mask=mask_input
        )
        assert (analysis.lens_imaging_data.mask == mask_from_function).all()

        # If no mask is suppled, nor a mask function, we should use the default mask. This extends behind the edge of
        # 5x5 image, so will raise a MaskException.

        phase_imaging_7x7.mask_function = None

        with pytest.raises(exc.MaskException):
            phase_imaging_7x7.make_analysis(data=imaging_data_7x7, mask=None)

    def test__make_analysis__mask_input_uses_mask__inner_mask_radius_included_which_masks_centre(
        self, phase_imaging_7x7, imaging_data_7x7
    ):
        # If an input mask is supplied and there is no mask function, we use mask input.

        phase_imaging_7x7.mask_function = None
        phase_imaging_7x7.inner_mask_radii = 0.5

        mask_input = al.Mask.circular(
            shape=imaging_data_7x7.shape, pixel_scale=1, sub_size=1, radius_arcsec=1.5
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, mask=mask_input
        )

        # The inner circulaar mask radii of 0.5" masks only the central pixel of the mask

        mask_input[3, 3] = True

        assert (analysis.lens_imaging_data.mask == mask_input).all()

        # If a mask function is supplied, we should use this mask, regardless of whether an input mask is supplied.

        def mask_function(image, sub_size):
            return al.Mask.circular(
                shape=image.shape, pixel_scale=1, sub_size=sub_size, radius_arcsec=1.4
            )

        mask_from_function = mask_function(image=imaging_data_7x7.image, sub_size=1)

        # The inner circulaar mask radii of 1.0" masks the centra pixels of the mask
        mask_from_function[3, 3] = True

        phase_imaging_7x7.mask_function = mask_function

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7, mask=None)
        assert (analysis.lens_imaging_data.mask == mask_from_function).all()

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, mask=mask_input
        )
        assert (analysis.lens_imaging_data.mask == mask_from_function).all()

        # If no mask is suppled, nor a mask function, we should use the default mask.

        phase_imaging_7x7.mask_function = None

        with pytest.raises(exc.MaskException):
            phase_imaging_7x7.make_analysis(data=imaging_data_7x7, mask=None)

    def test__make_analysis__mask_changes_sub_size_depending_on_phase_attribute(
        self, phase_imaging_7x7, imaging_data_7x7
    ):
        # If an input mask is supplied and there is no mask function, we use mask input.

        phase_imaging_7x7.mask_function = None

        mask_input = al.Mask.circular(
            shape=imaging_data_7x7.shape, pixel_scale=1, sub_size=1, radius_arcsec=1.5
        )

        phase_imaging_7x7.sub_size = 1
        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, mask=mask_input
        )

        assert (analysis.lens_imaging_data.mask == mask_input).all()
        assert analysis.lens_imaging_data.sub_size == 1

        phase_imaging_7x7.sub_size = 2
        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, mask=mask_input
        )

        assert (analysis.lens_imaging_data.mask == mask_input).all()
        assert analysis.lens_imaging_data.sub_size == 2

    def test__make_analysis__positions_are_input__are_used_in_analysis(
        self, phase_imaging_7x7, imaging_data_7x7
    ):
        # If position threshold is input (not None) and positions are input, make the positions part of the lens data.

        phase_imaging_7x7.positions_threshold = 0.2

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, positions=[[[1.0, 1.0], [2.0, 2.0]]]
        )

        assert (
            analysis.lens_imaging_data.positions[0][0] == np.array([1.0, 1.0])
        ).all()
        assert (
            analysis.lens_imaging_data.positions[0][1] == np.array([2.0, 2.0])
        ).all()
        assert analysis.lens_imaging_data.positions_threshold == 0.2

        # If position threshold is input (not None) and but no positions are supplied, raise an error

        with pytest.raises(exc.PhaseException):
            phase_imaging_7x7.make_analysis(data=imaging_data_7x7, positions=None)
            phase_imaging_7x7.make_analysis(data=imaging_data_7x7)

    def test__make_analysis__positions_do_not_trace_within_threshold__raises_exception(
        self, phase_imaging_7x7, imaging_data_7x7, mask_function_7x7
    ):
        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(source=al.Galaxy(redshift=0.5)),
            mask_function=mask_function_7x7,
            positions_threshold=50.0,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, positions=[[[1.0, 1.0], [2.0, 2.0]]]
        )
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.check_positions_trace_within_threshold_via_tracer(tracer=tracer)

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(source=al.Galaxy(redshift=0.5)),
            mask_function=mask_function_7x7,
            positions_threshold=0.0,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, positions=[[[1.0, 1.0], [2.0, 2.0]]]
        )
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.RayTracingException):
            analysis.check_positions_trace_within_threshold_via_tracer(tracer=tracer)

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(source=al.Galaxy(redshift=0.5)),
            mask_function=mask_function_7x7,
            positions_threshold=0.5,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, positions=[[[1.0, 0.0], [-1.0, 0.0]]]
        )
        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    mass=al.mass_profiles.SphericalIsothermal(einstein_radius=1.0),
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        analysis.check_positions_trace_within_threshold_via_tracer(tracer=tracer)

        tracer = al.Tracer.from_galaxies(
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    mass=al.mass_profiles.SphericalIsothermal(einstein_radius=0.0),
                ),
                al.Galaxy(redshift=1.0),
            ]
        )

        with pytest.raises(exc.RayTracingException):
            analysis.check_positions_trace_within_threshold_via_tracer(tracer=tracer)

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7,
            positions=[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        )
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.check_positions_trace_within_threshold_via_tracer(tracer=tracer)

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7,
            positions=[[[0.0, 0.0], [0.0, 0.0]], [[100.0, 0.0], [0.0, 0.0]]],
        )
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.RayTracingException):
            analysis.check_positions_trace_within_threshold_via_tracer(tracer=tracer)

    def test__make_analysis__inversion_resolution_error_raised_if_above_inversion_pixel_limit(
        self, phase_imaging_7x7, imaging_data_7x7, mask_function_7x7
    ):
        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                source=al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pixelizations.Rectangular(shape=(3, 3)),
                    regularization=al.regularization.Constant(),
                )
            ),
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)

        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.check_inversion_pixels_are_below_limit_via_tracer(tracer=tracer)

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                source=al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pixelizations.Rectangular(shape=(4, 4)),
                    regularization=al.regularization.Constant(),
                )
            ),
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.PixelizationException):
            analysis.check_inversion_pixels_are_below_limit_via_tracer(tracer=tracer)
            analysis.fit(instance=instance)

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                source=al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pixelizations.Rectangular(shape=(3, 3)),
                    regularization=al.regularization.Constant(),
                )
            ),
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.check_inversion_pixels_are_below_limit_via_tracer(tracer=tracer)

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=dict(
                source=al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pixelizations.Rectangular(shape=(4, 4)),
                    regularization=al.regularization.Constant(),
                )
            ),
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.PixelizationException):
            analysis.check_inversion_pixels_are_below_limit_via_tracer(tracer=tracer)
            analysis.fit(instance=instance)

    def test__make_analysis__pixel_scale_interpolation_grid_is_input__interp_grid_used_in_analysis(
        self, phase_imaging_7x7, imaging_data_7x7
    ):
        # If use positions is true and positions are input, make the positions part of the lens data.

        phase_imaging_7x7.pixel_scale_interpolation_grid = 0.1

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
        assert analysis.lens_imaging_data.pixel_scale_interpolation_grid == 0.1
        assert hasattr(analysis.lens_imaging_data.grid, "interpolator")
        assert hasattr(analysis.lens_imaging_data.blurring_grid, "interpolator")

    def test__make_analysis__inversion_pixel_limit__is_input__used_in_analysis(
        self, phase_imaging_7x7, imaging_data_7x7, mask_7x7
    ):
        phase_imaging_7x7.galaxies.lens = al.GalaxyModel(
            redshift=0.5,
            pixelization=al.pixelizations.VoronoiBrightnessImage,
            regularization=al.regularization.Constant,
        )

        phase_imaging_7x7.pixel_scale_binned_cluster_grid = mask_7x7.pixel_scale
        phase_imaging_7x7.inversion_pixel_limit = 5

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)

        assert (
            analysis.lens_imaging_data.pixel_scale_binned_grid == mask_7x7.pixel_scale
        )

        # There are 9 pixels in the mask, so to meet the inversoin pixel limit the pixel scale will be rescaled to the
        # masks's pixel scale

        phase_imaging_7x7.pixel_scale_binned_cluster_grid = mask_7x7.pixel_scale * 2.0
        phase_imaging_7x7.inversion_pixel_limit = 5

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)

        assert (
            analysis.lens_imaging_data.pixel_scale_binned_grid == mask_7x7.pixel_scale
        )

        # This image cannot meet the requirement, so will raise an error.

        phase_imaging_7x7.pixel_scale_binned_cluster_grid = mask_7x7.pixel_scale * 2.0
        phase_imaging_7x7.inversion_pixel_limit = 10

        with pytest.raises(exc.DataException):
            phase_imaging_7x7.make_analysis(data=imaging_data_7x7)

    def test__pixelization_property_extracts_pixelization(
        self, mask_function_7x7, imaging_data_7x7
    ):
        source_galaxy = al.Galaxy(redshift=0.5)

        phase_imaging_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[source_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        assert phase_imaging_7x7.pixelization == None

        source_galaxy = al.Galaxy(
            redshift=0.5,
            pixelization=al.pixelizations.Rectangular(),
            regularization=al.regularization.Constant(),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[source_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        assert isinstance(phase_imaging_7x7.pixelization, al.pixelizations.Rectangular)

        source_galaxy = al.GalaxyModel(
            redshift=0.5,
            pixelization=al.pixelizations.Rectangular,
            regularization=al.regularization.Constant,
        )

        phase_imaging_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[source_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        assert type(phase_imaging_7x7.pixelization) == type(
            al.pixelizations.Rectangular
        )

    def test__default_mask_function(self, phase_imaging_7x7, imaging_data_7x7):
        lens_data = al.LensImagingData(
            imaging_data=imaging_data_7x7,
            mask=phase_imaging_7x7.mask_function(
                image=imaging_data_7x7.image, sub_size=1
            ),
        )

        assert len(lens_data._image_1d) == 9

    def test__check_if_phase_uses_cluster_inversion(self, mask_function_7x7):
        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5), source=al.GalaxyModel(redshift=1.0)
            ),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is False

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.GalaxyModel(
                    redshift=0.5,
                    pixelization=al.pixelizations.Rectangular,
                    regularization=al.regularization.Constant,
                ),
                source=al.GalaxyModel(redshift=1.0),
            ),
        )
        assert phase_imaging_7x7.uses_cluster_inversion is False

        source = al.GalaxyModel(
            redshift=1.0,
            pixelization=al.pixelizations.VoronoiBrightnessImage,
            regularization=al.regularization.Constant,
        )

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(lens=al.GalaxyModel(redshift=0.5), source=source),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is True

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5), source=al.GalaxyModel(redshift=1.0)
            ),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is False

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.GalaxyModel(
                    redshift=0.5,
                    pixelization=al.pixelizations.Rectangular,
                    regularization=al.regularization.Constant,
                ),
                source=al.GalaxyModel(redshift=1.0),
            ),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is False

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5),
                source=al.GalaxyModel(
                    redshift=1.0,
                    pixelization=al.pixelizations.VoronoiBrightnessImage,
                    regularization=al.regularization.Constant,
                ),
            ),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is True

    def test__use_border__determines_if_border_pixel_relocation_is_used(
        self, imaging_data_7x7, mask_function_7x7, lens_imaging_data_7x7
    ):
        # noinspection PyTypeChecker

        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mass_profiles.SphericalIsothermal(einstein_radius=100.0),
        )
        source_galaxy = al.Galaxy(
            redshift=1.0,
            pixelization=al.pixelizations.Rectangular(shape=(3, 3)),
            regularization=al.regularization.Constant(coefficient=1.0),
        )

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy, source_galaxy],
            mask_function=mask_function_7x7,
            cosmology=cosmo.Planck15,
            phase_name="test_phase",
            inversion_uses_border=True,
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
        analysis.lens_imaging_data.grid[4] = np.array([[500.0, 0.0]])

        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.lens_imaging_fit_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.grid[4][0] == pytest.approx(97.19584, 1.0e-2)
        assert fit.inversion.mapper.grid[4][1] == pytest.approx(-3.699999, 1.0e-2)

        phase_imaging_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy, source_galaxy],
            mask_function=mask_function_7x7,
            cosmology=cosmo.Planck15,
            phase_name="test_phase",
            inversion_uses_border=False,
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)

        analysis.lens_imaging_data.grid[4] = np.array([300.0, 0.0])

        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.lens_imaging_fit_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.grid[4][0] == pytest.approx(200.0, 1.0e-4)

    def test__inversion_pixel_limit_computed_via_config_or_input(
        self, mask_function_7x7
    ):
        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_imaging_7x7",
            mask_function=mask_function_7x7,
            inversion_pixel_limit=None,
        )

        assert phase_imaging_7x7.inversion_pixel_limit == 3000

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_imaging_7x7",
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
        )

        assert phase_imaging_7x7.inversion_pixel_limit == 10

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_imaging_7x7",
            mask_function=mask_function_7x7,
            inversion_pixel_limit=2000,
        )

        assert phase_imaging_7x7.inversion_pixel_limit == 2000

    def test__make_analysis_determines_if_pixelization_is_same_as_previous_phase(
        self, imaging_data_7x7, mask_function_7x7, results_collection_7x7
    ):
        results_collection_7x7.last.hyper_combined.preload_pixelization_grids_of_planes = (
            1
        )

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase", mask_function=mask_function_7x7
        )

        results_collection_7x7.last.pixelization = None

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )

        assert analysis.lens_imaging_data.preload_pixelization_grids_of_planes is None

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase", mask_function=mask_function_7x7
        )

        results_collection_7x7.last.pixelization = al.pixelizations.Rectangular

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )

        assert analysis.lens_imaging_data.preload_pixelization_grids_of_planes is None

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pixelizations.Rectangular,
                    regularization=al.regularization.Constant,
                )
            ],
        )

        results_collection_7x7.last.pixelization = None

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )

        assert analysis.lens_imaging_data.preload_pixelization_grids_of_planes is None

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    pixelization=al.pixelizations.Rectangular,
                    regularization=al.regularization.Constant,
                )
            ],
        )

        results_collection_7x7.last.pixelization = al.pixelizations.Rectangular

        analysis = phase_imaging_7x7.make_analysis(
            data=imaging_data_7x7, results=results_collection_7x7
        )

        assert analysis.lens_imaging_data.preload_pixelization_grids_of_planes == 1

    #
    # def test__uses_pixelization_preload_grids_if_possible(
    #     self, imaging_data_7x7, mask_function_7x7
    # ):
    #     phase_imaging_7x7 = al.PhaseImaging(
    #         phase_name="test_phase", mask_function=mask_function_7x7
    #     )
    #
    #     analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
    #
    #     galaxy = al.Galaxy(redshift=0.5)
    #
    #     preload_pixelization_grid = analysis.setup_peload_pixelization_grid(
    #         galaxies=[galaxy, galaxy], grid=analysis.lens_imaging_data.grid
    #     )
    #
    #     assert (preload_pixelization_grid.pixelization == np.array([[0.0, 0.0]])).all()
    #
    #     galaxy_pix_which_doesnt_use_pix_grid = al.Galaxy(
    #         redshift=0.5, pixelization=al.pixelizations.Rectangular(), regularization=al.regularization.Constant()
    #     )
    #
    #     preload_pixelization_grid = analysis.setup_peload_pixelization_grid(
    #         galaxies=[galaxy_pix_which_doesnt_use_pix_grid],
    #         grid=analysis.lens_imaging_data.grid,
    #     )
    #
    #     assert (preload_pixelization_grid.pixelization == np.array([[0.0, 0.0]])).all()
    #
    #     galaxy_pix_which_uses_pix_grid = al.Galaxy(
    #         redshift=0.5,
    #         pixelization=al.pixelizations.VoronoiMagnification(),
    #         regularization=al.regularization.Constant(),
    #     )
    #
    #     preload_pixelization_grid = analysis.setup_peload_pixelization_grid(
    #         galaxies=[galaxy_pix_which_uses_pix_grid],
    #         grid=analysis.lens_imaging_data.grid,
    #     )
    #
    #     assert (
    #         preload_pixelization_grid.pixelization
    #         == np.array(
    #             [
    #                 [1.0, -1.0],
    #                 [1.0, 0.0],
    #                 [1.0, 1.0],
    #                 [0.0, -1.0],
    #                 [0.0, 0.0],
    #                 [0.0, 1.0],
    #                 [-1.0, -1.0],
    #                 [-1.0, 0.0],
    #                 [-1.0, 1.0],
    #             ]
    #         )
    #     ).all()
    #
    #     galaxy_pix_which_uses_brightness = al.Galaxy(
    #         redshift=0.5,
    #         pixelization=al.pixelizations.VoronoiBrightnessImage(pixels=9),
    #         regularization=al.regularization.Constant(),
    #     )
    #
    #     galaxy_pix_which_uses_brightness.hyper_galaxy_cluster_image_1d = np.array(
    #         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    #     )
    #
    #     phase_imaging_7x7 = al.PhaseImaging(
    #         phase_name="test_phase",
    #         galaxies=dict(
    #             lens=al.GalaxyModel(
    #                 redshift=0.5,
    #                 pixelization=al.pixelizations.VoronoiBrightnessImage,
    #                 regularization=al.regularization.Constant,
    #             )
    #         ),
    #         inversion_pixel_limit=5,
    #         mask_function=mask_function_7x7,
    #     )
    #
    #     analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
    #
    #     preload_pixelization_grid = analysis.setup_peload_pixelization_grid(
    #         galaxies=[galaxy_pix_which_uses_brightness],
    #         grid=analysis.lens_imaging_data.grid,
    #     )
    #
    #     assert (
    #         preload_pixelization_grid.pixelization
    #         == np.array(
    #             [
    #                 [0.0, 1.0],
    #                 [1.0, -1.0],
    #                 [-1.0, -1.0],
    #                 [-1.0, 1.0],
    #                 [0.0, -1.0],
    #                 [1.0, 1.0],
    #                 [-1.0, 0.0],
    #                 [0.0, 0.0],
    #                 [1.0, 0.0],
    #             ]
    #         )
    #     ).all()


class TestResult(object):
    def test__results_of_phase_are_available_as_properties(
        self, imaging_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    light=al.light_profiles.EllipticalSersic(intensity=1.0),
                )
            ],
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(data=imaging_data_7x7)

        assert isinstance(result, al.AbstractPhase.Result)

    def test__results_of_phase_include_mask__available_as_property(
        self, imaging_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    light=al.light_profiles.EllipticalSersic(intensity=1.0),
                )
            ],
            sub_size=2,
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(data=imaging_data_7x7)

        mask = mask_function_7x7(image=imaging_data_7x7.image, sub_size=2)

        assert (result.mask == mask).all()

    def test__results_of_phase_include_positions__available_as_property(
        self, imaging_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    light=al.light_profiles.EllipticalSersic(intensity=1.0),
                )
            ],
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(data=imaging_data_7x7)

        assert result.positions == None

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5,
                    light=al.light_profiles.EllipticalSersic(intensity=1.0),
                ),
                source=al.Galaxy(redshift=1.0),
            ),
            positions_threshold=1.0,
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(data=imaging_data_7x7, positions=[[[1.0, 1.0]]])

        assert (result.positions[0] == np.array([1.0, 1.0])).all()

    def test__results_of_phase_include_pixelization__available_as_property(
        self, imaging_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5,
                    light=al.light_profiles.EllipticalSersic(intensity=1.0),
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    pixelization=al.pixelizations.VoronoiMagnification(shape=(2, 3)),
                    regularization=al.regularization.Constant(),
                ),
            ),
            inversion_pixel_limit=6,
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(data=imaging_data_7x7)

        assert isinstance(result.pixelization, al.pixelizations.VoronoiMagnification)
        assert result.pixelization.shape == (2, 3)

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5,
                    light=al.light_profiles.EllipticalSersic(intensity=1.0),
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    pixelization=al.pixelizations.VoronoiBrightnessImage(pixels=6),
                    regularization=al.regularization.Constant(),
                ),
            ),
            inversion_pixel_limit=6,
            phase_name="test_phase_2",
        )

        phase_imaging_7x7.galaxies.source.binned_hyper_galaxy_image_1d = np.ones(9)

        result = phase_imaging_7x7.run(data=imaging_data_7x7)

        assert isinstance(result.pixelization, al.pixelizations.VoronoiBrightnessImage)
        assert result.pixelization.pixels == 6

    def test__results_of_phase_include_pixelization_grid__available_as_property(
        self, imaging_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    light=al.light_profiles.EllipticalSersic(intensity=1.0),
                )
            ],
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(data=imaging_data_7x7)

        assert result.most_likely_pixelization_grids_of_planes == None

        phase_imaging_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.Galaxy(
                    redshift=0.5,
                    light=al.light_profiles.EllipticalSersic(intensity=1.0),
                ),
                source=al.Galaxy(
                    redshift=1.0,
                    pixelization=al.pixelizations.VoronoiBrightnessImage(pixels=6),
                    regularization=al.regularization.Constant(),
                ),
            ),
            inversion_pixel_limit=6,
            phase_name="test_phase_2",
        )

        phase_imaging_7x7.galaxies.source.binned_hyper_galaxy_image_1d = np.ones(9)

        result = phase_imaging_7x7.run(data=imaging_data_7x7)

        assert result.most_likely_pixelization_grids_of_planes.shape == (6, 2)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, imaging_data_7x7, mask_function_7x7
    ):
        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.light_profiles.EllipticalSersic(intensity=0.1)
        )

        phase_imaging_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[lens_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase_imaging_7x7.mask_function(image=imaging_data_7x7.image, sub_size=2)
        lens_data = al.LensImagingData(imaging_data=imaging_data_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = al.LensImagingFit.from_lens_data_and_tracer(
            lens_data=lens_data, tracer=tracer
        )

        assert fit.likelihood == fit_figure_of_merit

    def test__fit_figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, imaging_data_7x7, mask_function_7x7
    ):
        hyper_image_sky = al.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = al.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.light_profiles.EllipticalSersic(intensity=0.1)
        )

        phase_imaging_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[lens_galaxy],
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(data=imaging_data_7x7)
        instance = phase_imaging_7x7.variable.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase_imaging_7x7.mask_function(image=imaging_data_7x7.image, sub_size=2)
        lens_data = al.LensImagingData(imaging_data=imaging_data_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = al.LensImagingFit.from_lens_data_and_tracer(
            lens_data=lens_data,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.likelihood == fit_figure_of_merit


class TestPhasePickle(object):

    # noinspection PyTypeChecker
    def test_assertion_failure(self, imaging_data_7x7, mask_function_7x7):
        def make_analysis(*args, **kwargs):
            return mock_pipeline.MockAnalysis(1, 1)

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    light=al.light_profiles.EllipticalLightProfile, redshift=1
                )
            ),
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(
            data=imaging_data_7x7, results=None, mask=None, positions=None
        )
        assert result is not None

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    light=al.light_profiles.EllipticalLightProfile, redshift=1
                )
            ),
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(
            data=imaging_data_7x7, results=None, mask=None, positions=None
        )
        assert result is not None

        class CustomPhase(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies.lens.light = al.light_profiles.EllipticalLightProfile()

        phase_imaging_7x7 = CustomPhase(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    light=al.light_profiles.EllipticalLightProfile, redshift=1
                )
            ),
        )
        phase_imaging_7x7.make_analysis = make_analysis

        # with pytest.raises(af.exc.PipelineException):
        #     phase_imaging_7x7.run(data_type=imaging_data_7x7, results=None, mask=None, positions=None)
