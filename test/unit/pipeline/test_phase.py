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
        "{}/../test_files/config/phase_7x7".format(directory)
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
    def test_set_constants(self, phase_7x7):
        phase_7x7.galaxies = [al.Galaxy(redshift=0.5)]
        assert phase_7x7.optimizer.variable.galaxies == [al.Galaxy(redshift=0.5)]

    def test_set_variables(self, phase_7x7):
        phase_7x7.galaxies = [al.GalaxyModel(redshift=0.5)]
        assert phase_7x7.optimizer.variable.galaxies == [al.GalaxyModel(redshift=0.5)]

    def test__make_analysis(self, phase_7x7, ccd_data_7x7, lens_data_7x7):
        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        assert analysis.last_results is None
        assert analysis.lens_data.unmasked_image == ccd_data_7x7.image
        assert analysis.lens_data.unmasked_noise_map == ccd_data_7x7.noise_map
        assert analysis.lens_data.image(return_in_2d=True) == lens_data_7x7.image(
            return_in_2d=True
        )
        assert analysis.lens_data.noise_map(
            return_in_2d=True
        ) == lens_data_7x7.noise_map(return_in_2d=True)

    def test_make_analysis__mask_input_uses_mask__no_mask_uses_mask_function(
        self, phase_7x7, ccd_data_7x7
    ):
        # If an input mask is supplied and there is no mask function, we use mask input.

        phase_7x7.mask_function = None

        mask_input = al.Mask.circular(
            shape=ccd_data_7x7.shape, pixel_scale=1, radius_arcsec=1.5
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7, mask=mask_input)

        assert (analysis.lens_data.mask_2d == mask_input).all()

        # If a mask function is suppled, we should use this mask, regardless of whether an input mask is supplied.

        def mask_function(image):
            return al.Mask.circular(shape=image.shape, pixel_scale=1, radius_arcsec=0.3)

        mask_from_function = mask_function(image=ccd_data_7x7.image)
        phase_7x7.mask_function = mask_function

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7, mask=None)
        assert (analysis.lens_data.mask_2d == mask_from_function).all()
        analysis = phase_7x7.make_analysis(data=ccd_data_7x7, mask=mask_input)
        assert (analysis.lens_data.mask_2d == mask_from_function).all()

        # If no mask is suppled, nor a mask function, we should use the default mask. This extends behind the edge of
        # 5x5 image, so will raise a MaskException.

        phase_7x7.mask_function = None

        with pytest.raises(exc.MaskException):
            phase_7x7.make_analysis(data=ccd_data_7x7, mask=None)

    def test__make_analysis__mask_input_uses_mask__inner_mask_radius_included_which_masks_centre(
        self, phase_7x7, ccd_data_7x7
    ):
        # If an input mask is supplied and there is no mask function, we use mask input.

        phase_7x7.mask_function = None
        phase_7x7.inner_mask_radii = 0.5

        mask_input = al.Mask.circular(
            shape=ccd_data_7x7.shape, pixel_scale=1, radius_arcsec=1.5
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7, mask=mask_input)

        # The inner circulaar mask radii of 0.5" masks only the central pixel of the mask

        mask_input[3, 3] = True

        assert (analysis.lens_data.mask_2d == mask_input).all()

        # If a mask function is supplied, we should use this mask, regardless of whether an input mask is supplied.

        def mask_function(image):
            return al.Mask.circular(shape=image.shape, pixel_scale=1, radius_arcsec=1.4)

        mask_from_function = mask_function(image=ccd_data_7x7.image)

        # The inner circulaar mask radii of 1.0" masks the centra pixels of the mask
        mask_from_function[3, 3] = True

        phase_7x7.mask_function = mask_function

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7, mask=None)
        assert (analysis.lens_data.mask_2d == mask_from_function).all()

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7, mask=mask_input)
        assert (analysis.lens_data.mask_2d == mask_from_function).all()

        # If no mask is suppled, nor a mask function, we should use the default mask.

        phase_7x7.mask_function = None

        with pytest.raises(exc.MaskException):
            phase_7x7.make_analysis(data=ccd_data_7x7, mask=None)

    def test__make_analysis__positions_are_input__are_used_in_analysis(
        self, phase_7x7, ccd_data_7x7
    ):
        # If position threshold is input (not None) and positions are input, make the positions part of the lens data.

        phase_7x7.positions_threshold = 0.2

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7, positions=[[[1.0, 1.0], [2.0, 2.0]]]
        )

        assert (analysis.lens_data.positions[0][0] == np.array([1.0, 1.0])).all()
        assert (analysis.lens_data.positions[0][1] == np.array([2.0, 2.0])).all()
        assert analysis.lens_data.positions_threshold == 0.2

        # If position threshold is input (not None) and but no positions are supplied, raise an error

        with pytest.raises(exc.PhaseException):
            phase_7x7.make_analysis(data=ccd_data_7x7, positions=None)
            phase_7x7.make_analysis(data=ccd_data_7x7)

    def test__make_analysis__positions_do_not_trace_within_threshold__raises_exception(
        self, phase_7x7, ccd_data_7x7, mask_function_7x7
    ):
        phase_7x7 = al.PhaseImaging(
            galaxies=dict(source=al.Galaxy(redshift=0.5)),
            mask_function=mask_function_7x7,
            positions_threshold=50.0,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7, positions=[[[1.0, 1.0], [2.0, 2.0]]]
        )
        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.check_positions_trace_within_threshold_via_tracer(tracer=tracer)

        phase_7x7 = al.PhaseImaging(
            galaxies=dict(source=al.Galaxy(redshift=0.5)),
            mask_function=mask_function_7x7,
            positions_threshold=0.0,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7, positions=[[[1.0, 1.0], [2.0, 2.0]]]
        )
        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.RayTracingException):
            analysis.check_positions_trace_within_threshold_via_tracer(tracer=tracer)

        phase_7x7 = al.PhaseImaging(
            galaxies=dict(source=al.Galaxy(redshift=0.5)),
            mask_function=mask_function_7x7,
            positions_threshold=0.5,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7, positions=[[[1.0, 0.0], [-1.0, 0.0]]]
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

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7,
            positions=[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        )
        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.check_positions_trace_within_threshold_via_tracer(tracer=tracer)

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7,
            positions=[[[0.0, 0.0], [0.0, 0.0]], [[100.0, 0.0], [0.0, 0.0]]],
        )
        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.RayTracingException):
            analysis.check_positions_trace_within_threshold_via_tracer(tracer=tracer)

    def test__make_analysis__inversion_resolution_error_raised_if_above_inversion_pixel_limit(
        self, phase_7x7, ccd_data_7x7, mask_function_7x7
    ):
        phase_7x7 = al.PhaseImaging(
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

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.check_inversion_pixels_are_below_limit_via_tracer(tracer=tracer)

        phase_7x7 = al.PhaseImaging(
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

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.PixelizationException):
            analysis.check_inversion_pixels_are_below_limit_via_tracer(tracer=tracer)
            analysis.fit(instance=instance)

        phase_7x7 = al.PhaseImaging(
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

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        analysis.check_inversion_pixels_are_below_limit_via_tracer(tracer=tracer)

        phase_7x7 = al.PhaseImaging(
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

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        with pytest.raises(exc.PixelizationException):
            analysis.check_inversion_pixels_are_below_limit_via_tracer(tracer=tracer)
            analysis.fit(instance=instance)

    def test_make_analysis__pixel_scale_interpolation_grid_is_input__interp_grid_used_in_analysis(
        self, phase_7x7, ccd_data_7x7
    ):
        # If use positions is true and positions are input, make the positions part of the lens data.

        phase_7x7.pixel_scale_interpolation_grid = 0.1

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        assert analysis.lens_data.pixel_scale_interpolation_grid == 0.1
        assert hasattr(analysis.lens_data.grid, "interpolator")
        assert hasattr(analysis.lens_data.preload_blurring_grid, "interpolator")

    def test_make_analysis__inversion_pixel_limit__is_input__used_in_analysis(
        self, phase_7x7, ccd_data_7x7, mask_7x7
    ):
        phase_7x7.galaxies.lens = al.GalaxyModel(
            redshift=0.5,
            pixelization=al.pixelizations.VoronoiBrightnessImage,
            regularization=al.regularization.Constant,
        )

        phase_7x7.pixel_scale_binned_cluster_grid = mask_7x7.pixel_scale
        phase_7x7.inversion_pixel_limit = 5

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        assert analysis.lens_data.pixel_scale_binned_grid == mask_7x7.pixel_scale

        # There are 9 pixels in the mask, so to meet the inversoin pixel limit the pixel scale will be rescaled to the
        # masks's pixel scale

        phase_7x7.pixel_scale_binned_cluster_grid = mask_7x7.pixel_scale * 2.0
        phase_7x7.inversion_pixel_limit = 5

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        assert analysis.lens_data.pixel_scale_binned_grid == mask_7x7.pixel_scale

        # This image cannot meet the requirement, so will raise an error.

        phase_7x7.pixel_scale_binned_cluster_grid = mask_7x7.pixel_scale * 2.0
        phase_7x7.inversion_pixel_limit = 10

        with pytest.raises(exc.DataException):
            phase_7x7.make_analysis(data=ccd_data_7x7)

    def test__make_analysis__phase_info_is_made(self, phase_7x7, ccd_data_7x7):
        phase_7x7.make_analysis(data=ccd_data_7x7)

        file_phase_info = "{}/{}".format(
            phase_7x7.optimizer.phase_output_path, "phase.info"
        )

        phase_info = open(file_phase_info, "r")

        optimizer = phase_info.readline()
        sub_grid_size = phase_info.readline()
        psf_shape = phase_info.readline()
        positions_threshold = phase_info.readline()
        cosmology = phase_info.readline()
        auto_link_priors = phase_info.readline()

        phase_info.close()

        assert optimizer == "Optimizer = MockNLO \n"
        assert sub_grid_size == "Sub-grid size = 2 \n"
        assert psf_shape == "PSF shape = None \n"
        assert positions_threshold == "Positions Threshold = None \n"
        assert (
            cosmology
            == 'Cosmology = FlatLambdaCDM(name="Planck15", H0=67.7 km / (Mpc s), Om0=0.307, Tcmb0=2.725 K, '
            "Neff=3.05, m_nu=[0.   0.   0.06] eV, Ob0=0.0486) \n"
        )
        assert auto_link_priors == "Auto Link Priors = False \n"

    def test_pixelization_property_extracts_pixelization(
        self, mask_function_7x7, ccd_data_7x7
    ):
        source_galaxy = al.Galaxy(redshift=0.5)

        phase_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[source_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        assert phase_7x7.pixelization == None

        source_galaxy = al.Galaxy(
            redshift=0.5,
            pixelization=al.pixelizations.Rectangular(),
            regularization=al.regularization.Constant(),
        )

        phase_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[source_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        assert isinstance(phase_7x7.pixelization, al.pixelizations.Rectangular)

        source_galaxy = al.GalaxyModel(
            redshift=0.5,
            pixelization=al.pixelizations.Rectangular,
            regularization=al.regularization.Constant,
        )

        phase_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[source_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        assert type(phase_7x7.pixelization) == type(al.pixelizations.Rectangular)

    def test_fit(self, ccd_data_7x7, mask_function_7x7):
        clean_images()

        phase_7x7 = al.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.GalaxyModel(
                    redshift=0.5, light=al.light_profiles.EllipticalSersic
                ),
                source=al.GalaxyModel(
                    redshift=1.0, light=al.light_profiles.EllipticalSersic
                ),
            ),
            mask_function=mask_function_7x7,
            phase_name="test_phase_test_fit",
        )

        result = phase_7x7.run(data=ccd_data_7x7)
        assert isinstance(result.constant.galaxies[0], al.Galaxy)
        assert isinstance(result.constant.galaxies[0], al.Galaxy)

    def test_customize(
        self, mask_function_7x7, results_7x7, results_collection_7x7, ccd_data_7x7
    ):
        class MyPlanePhaseAnd(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies = results.last.constant.galaxies

        galaxy = al.Galaxy(redshift=0.5)
        galaxy_model = al.GalaxyModel(redshift=0.5)

        setattr(results_7x7.constant, "galaxies", [galaxy])
        setattr(results_7x7.variable, "galaxies", [galaxy_model])

        phase_7x7 = MyPlanePhaseAnd(
            phase_name="test_phase",
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
        )

        phase_7x7.make_analysis(data=ccd_data_7x7, results=results_collection_7x7)
        phase_7x7.customize_priors(results_collection_7x7)

        assert phase_7x7.galaxies == [galaxy]

        class MyPlanePhaseAnd(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies = results.last.variable.galaxies

        galaxy = al.Galaxy(redshift=0.5)
        galaxy_model = al.GalaxyModel(redshift=0.5)

        setattr(results_7x7.constant, "galaxies", [galaxy])
        setattr(results_7x7.variable, "galaxies", [galaxy_model])

        phase_7x7 = MyPlanePhaseAnd(
            phase_name="test_phase",
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
        )

        phase_7x7.make_analysis(data=ccd_data_7x7, results=results_collection_7x7)
        phase_7x7.customize_priors(results_collection_7x7)

        assert phase_7x7.galaxies == [galaxy_model]

    def test_default_mask_function(self, phase_7x7, ccd_data_7x7):
        lens_data = al.LensData(
            ccd_data=ccd_data_7x7, mask=phase_7x7.mask_function(ccd_data_7x7.image)
        )

        assert len(lens_data.image_1d) == 9

    def test_duplication(self):
        phase_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5), source=al.GalaxyModel(redshift=1.0)
            ),
        )

        al.PhaseImaging(phase_name="test_phase")

        assert phase_7x7.galaxies is not None

    def test_modify_image(self, mask_function_7x7, ccd_data_7x7):
        class MyPhase(al.PhaseImaging):
            def modify_image(self, image, results):
                assert ccd_data_7x7.image.shape == image.shape
                image = 20.0 * np.ones(shape=(5, 5))
                return image

        phase_7x7 = MyPhase(phase_name="phase_7x7", mask_function=mask_function_7x7)

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        assert (analysis.lens_data.unmasked_image == 20.0 * np.ones(shape=(5, 5))).all()
        assert (analysis.lens_data.image_1d == 20.0 * np.ones(shape=9)).all()

    def test__check_if_phase_uses_cluster_inversion(self, mask_function_7x7):
        phase_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5), source=al.GalaxyModel(redshift=1.0)
            ),
        )

        assert phase_7x7.uses_cluster_inversion is False

        phase_7x7 = al.PhaseImaging(
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
        assert phase_7x7.uses_cluster_inversion is False

        source = al.GalaxyModel(
            redshift=1.0,
            pixelization=al.pixelizations.VoronoiBrightnessImage,
            regularization=al.regularization.Constant,
        )

        phase_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(lens=al.GalaxyModel(redshift=0.5), source=source),
        )

        assert phase_7x7.uses_cluster_inversion is True

        phase_7x7 = al.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=al.GalaxyModel(redshift=0.5), source=al.GalaxyModel(redshift=1.0)
            ),
        )

        assert phase_7x7.uses_cluster_inversion is False

        phase_7x7 = al.PhaseImaging(
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

        assert phase_7x7.uses_cluster_inversion is False

        phase_7x7 = al.PhaseImaging(
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

        assert phase_7x7.uses_cluster_inversion is True

    def test__use_border__determines_if_border_pixel_relocation_is_used(
        self, ccd_data_7x7, mask_function_7x7, lens_data_7x7
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

        phase_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy, source_galaxy],
            mask_function=mask_function_7x7,
            cosmology=cosmo.Planck15,
            phase_name="test_phase",
            inversion_uses_border=True,
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        analysis.lens_data.grid[4] = np.array([[500.0, 0.0]])

        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.fit_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.grid[4][0] == pytest.approx(97.19584, 1.0e-2)
        assert fit.inversion.mapper.grid[4][1] == pytest.approx(-3.699999, 1.0e-2)

        phase_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy, source_galaxy],
            mask_function=mask_function_7x7,
            cosmology=cosmo.Planck15,
            phase_name="test_phase",
            inversion_uses_border=False,
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        analysis.lens_data.grid[4] = np.array([300.0, 0.0])

        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = analysis.fit_for_tracer(
            tracer=tracer, hyper_image_sky=None, hyper_background_noise=None
        )

        assert fit.inversion.mapper.grid[4][0] == pytest.approx(200.0, 1.0e-4)

    def test__inversion_pixel_limit_computed_via_config_or_input(
        self, mask_function_7x7
    ):
        phase_7x7 = al.PhaseImaging(
            phase_name="phase_7x7",
            mask_function=mask_function_7x7,
            inversion_pixel_limit=None,
        )

        assert phase_7x7.inversion_pixel_limit == 3000

        phase_7x7 = al.PhaseImaging(
            phase_name="phase_7x7",
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
        )

        assert phase_7x7.inversion_pixel_limit == 10

        phase_7x7 = al.PhaseImaging(
            phase_name="phase_7x7",
            mask_function=mask_function_7x7,
            inversion_pixel_limit=2000,
        )

        assert phase_7x7.inversion_pixel_limit == 2000

    def test__make_analysis_determines_if_pixelization_is_same_as_previous_phas(
        self, ccd_data_7x7, mask_function_7x7, results_collection_7x7
    ):
        results_collection_7x7.last.hyper_combined.preload_pixelization_grids_of_planes = (
            1
        )

        phase_7x7 = al.PhaseImaging(
            phase_name="test_phase", mask_function=mask_function_7x7
        )

        results_collection_7x7.last.pixelization = None

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7, results=results_collection_7x7
        )

        assert analysis.lens_data.preload_pixelization_grids_of_planes is None

        phase_7x7 = al.PhaseImaging(
            phase_name="test_phase", mask_function=mask_function_7x7
        )

        results_collection_7x7.last.pixelization = al.pixelizations.Rectangular

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7, results=results_collection_7x7
        )

        assert analysis.lens_data.preload_pixelization_grids_of_planes is None

        phase_7x7 = al.PhaseImaging(
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

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7, results=results_collection_7x7
        )

        assert analysis.lens_data.preload_pixelization_grids_of_planes is None

        phase_7x7 = al.PhaseImaging(
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

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7, results=results_collection_7x7
        )

        assert analysis.lens_data.preload_pixelization_grids_of_planes == 1

    #
    # def test__uses_pixelization_preload_grids_if_possible(
    #     self, ccd_data_7x7, mask_function_7x7
    # ):
    #     phase_7x7 = al.PhaseImaging(
    #         phase_name="test_phase", mask_function=mask_function_7x7
    #     )
    #
    #     analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
    #
    #     galaxy = al.Galaxy(redshift=0.5)
    #
    #     preload_pixelization_grid = analysis.setup_peload_pixelization_grid(
    #         galaxies=[galaxy, galaxy], grid=analysis.lens_data.grid
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
    #         grid=analysis.lens_data.grid,
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
    #         grid=analysis.lens_data.grid,
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
    #     phase_7x7 = al.PhaseImaging(
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
    #     analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
    #
    #     preload_pixelization_grid = analysis.setup_peload_pixelization_grid(
    #         galaxies=[galaxy_pix_which_uses_brightness],
    #         grid=analysis.lens_data.grid,
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

    def test__lens_data_signal_to_noise_limit(
        self, ccd_data_7x7, mask_7x7_1_pix, mask_function_7x7_1_pix
    ):
        ccd_data_snr_limit = ccd_data_7x7.new_ccd_data_with_signal_to_noise_limit(
            signal_to_noise_limit=1.0
        )

        phase_7x7 = al.PhaseImaging(
            phase_name="phase_7x7",
            signal_to_noise_limit=1.0,
            mask_function=mask_function_7x7_1_pix,
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        assert (analysis.lens_data.unmasked_image == ccd_data_snr_limit.image).all()
        assert (
            analysis.lens_data.unmasked_noise_map == ccd_data_snr_limit.noise_map
        ).all()

        ccd_data_snr_limit = ccd_data_7x7.new_ccd_data_with_signal_to_noise_limit(
            signal_to_noise_limit=0.1
        )

        phase_7x7 = al.PhaseImaging(
            phase_name="phase_7x7",
            signal_to_noise_limit=0.1,
            mask_function=mask_function_7x7_1_pix,
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        assert (analysis.lens_data.unmasked_image == ccd_data_snr_limit.image).all()
        assert (
            analysis.lens_data.unmasked_noise_map == ccd_data_snr_limit.noise_map
        ).all()

    def test__lens_data_is_binned_up(
        self, ccd_data_7x7, mask_7x7_1_pix, mask_function_7x7_1_pix
    ):
        binned_up_ccd_data = ccd_data_7x7.new_ccd_data_with_binned_up_arrays(
            bin_up_factor=2
        )

        binned_up_mask = mask_7x7_1_pix.binned_up_mask_from_mask(bin_up_factor=2)

        phase_7x7 = al.PhaseImaging(
            phase_name="phase_7x7",
            bin_up_factor=2,
            mask_function=mask_function_7x7_1_pix,
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        assert (analysis.lens_data.unmasked_image == binned_up_ccd_data.image).all()
        assert (analysis.lens_data.psf == binned_up_ccd_data.psf).all()
        assert (
            analysis.lens_data.unmasked_noise_map == binned_up_ccd_data.noise_map
        ).all()

        assert (analysis.lens_data.mask_2d == binned_up_mask).all()

        lens_data = al.LensData(ccd_data=ccd_data_7x7, mask=mask_7x7_1_pix)

        binned_up_lens_data = lens_data.new_lens_data_with_binned_up_ccd_data_and_mask(
            bin_up_factor=2
        )

        assert (
            analysis.lens_data.image(return_in_2d=True)
            == binned_up_lens_data.image(return_in_2d=True)
        ).all()
        assert (analysis.lens_data.psf == binned_up_lens_data.psf).all()
        assert (
            analysis.lens_data.noise_map(return_in_2d=True)
            == binned_up_lens_data.noise_map(return_in_2d=True)
        ).all()

        assert (analysis.lens_data.mask_2d == binned_up_lens_data.mask_2d).all()

        assert (analysis.lens_data.image_1d == binned_up_lens_data.image_1d).all()
        assert (
            analysis.lens_data.noise_map_1d == binned_up_lens_data.noise_map_1d
        ).all()

    def test__tracer_for_instance__includes_cosmology(
        self, ccd_data_7x7, mask_function_7x7
    ):
        lens_galaxy = al.Galaxy(redshift=0.5)
        source_galaxy = al.Galaxy(redshift=0.5)

        phase_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[lens_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)

        assert tracer.image_plane.galaxies[0] == lens_galaxy
        assert tracer.cosmology == cosmo.FLRW

        phase_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[lens_galaxy, source_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(ccd_data_7x7)
        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance)

        assert tracer.image_plane.galaxies[0] == lens_galaxy
        assert tracer.source_plane.galaxies[0] == source_galaxy
        assert tracer.cosmology == cosmo.FLRW

        galaxy_0 = al.Galaxy(redshift=0.1)
        galaxy_1 = al.Galaxy(redshift=0.2)
        galaxy_2 = al.Galaxy(redshift=0.3)

        phase_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[galaxy_0, galaxy_1, galaxy_2],
            cosmology=cosmo.WMAP7,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance)

        assert tracer.planes[0].galaxies[0] == galaxy_0
        assert tracer.planes[1].galaxies[0] == galaxy_1
        assert tracer.planes[2].galaxies[0] == galaxy_2
        assert tracer.cosmology == cosmo.WMAP7

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, ccd_data_7x7, mask_function_7x7
    ):
        # noinspection PyTypeChecker

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.light_profiles.EllipticalSersic(intensity=0.1)
        )

        phase_7x7 = al.PhaseImaging(
            galaxies=[lens_galaxy],
            mask_function=mask_function_7x7,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        instance = phase_7x7.variable.instance_from_unit_vector([])

        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase_7x7.mask_function(image=ccd_data_7x7.image)
        lens_data = al.LensData(ccd_data=ccd_data_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = al.LensProfileFit(lens_data=lens_data, tracer=tracer)

        assert fit.likelihood == fit_figure_of_merit

    def test__phase_can_receive_list_of_galaxy_models(self):
        phase_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(
                    sersic=al.light_profiles.EllipticalSersic,
                    sis=al.mass_profiles.SphericalIsothermal,
                    redshift=al.Redshift,
                ),
                lens1=al.GalaxyModel(
                    sis=al.mass_profiles.SphericalIsothermal, redshift=al.Redshift
                ),
            ),
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        for item in phase_7x7.variable.path_priors_tuples:
            print(item)

        sersic = phase_7x7.variable.galaxies[0].sersic
        sis = phase_7x7.variable.galaxies[0].sis
        lens_1_sis = phase_7x7.variable.galaxies[1].sis

        arguments = {
            sersic.centre[0]: 0.2,
            sersic.centre[1]: 0.2,
            sersic.axis_ratio: 0.0,
            sersic.phi: 0.1,
            sersic.effective_radius.priors[0]: 0.2,
            sersic.sersic_index: 0.6,
            sersic.intensity.priors[0]: 0.6,
            sis.centre[0]: 0.1,
            sis.centre[1]: 0.2,
            sis.einstein_radius.priors[0]: 0.3,
            phase_7x7.variable.galaxies[0].redshift.priors[0]: 0.4,
            lens_1_sis.centre[0]: 0.6,
            lens_1_sis.centre[1]: 0.5,
            lens_1_sis.einstein_radius.priors[0]: 0.7,
            phase_7x7.variable.galaxies[1].redshift.priors[0]: 0.8,
        }

        instance = phase_7x7.optimizer.variable.instance_for_arguments(
            arguments=arguments
        )

        assert instance.galaxies[0].sersic.centre[0] == 0.2
        assert instance.galaxies[0].sis.centre[0] == 0.1
        assert instance.galaxies[0].sis.centre[1] == 0.2
        assert instance.galaxies[0].sis.einstein_radius == 0.3
        assert instance.galaxies[0].redshift == 0.4
        assert instance.galaxies[1].sis.centre[0] == 0.6
        assert instance.galaxies[1].sis.centre[1] == 0.5
        assert instance.galaxies[1].sis.einstein_radius == 0.7
        assert instance.galaxies[1].redshift == 0.8

        class LensPlanePhase2(al.PhaseImaging):
            # noinspection PyUnusedLocal
            def pass_models(self, results):
                self.galaxies[0].sis.einstein_radius = 10.0

        phase_7x7 = LensPlanePhase2(
            galaxies=dict(
                lens=al.GalaxyModel(
                    sersic=al.light_profiles.EllipticalSersic,
                    sis=al.mass_profiles.SphericalIsothermal,
                    redshift=al.Redshift,
                ),
                lens1=al.GalaxyModel(
                    sis=al.mass_profiles.SphericalIsothermal, redshift=al.Redshift
                ),
            ),
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        # noinspection PyTypeChecker
        phase_7x7.pass_models(None)

        sersic = phase_7x7.variable.galaxies[0].sersic
        sis = phase_7x7.variable.galaxies[0].sis
        lens_1_sis = phase_7x7.variable.galaxies[1].sis

        arguments = {
            sersic.centre[0]: 0.01,
            sersic.centre[1]: 0.2,
            sersic.axis_ratio: 0.0,
            sersic.phi: 0.1,
            sersic.effective_radius.priors[0]: 0.2,
            sersic.sersic_index: 0.6,
            sersic.intensity.priors[0]: 0.6,
            sis.centre[0]: 0.1,
            sis.centre[1]: 0.2,
            phase_7x7.variable.galaxies[0].redshift.priors[0]: 0.4,
            lens_1_sis.centre[0]: 0.6,
            lens_1_sis.centre[1]: 0.5,
            lens_1_sis.einstein_radius.priors[0]: 0.7,
            phase_7x7.variable.galaxies[1].redshift.priors[0]: 0.8,
        }

        instance = phase_7x7.optimizer.variable.instance_for_arguments(arguments)

        assert instance.galaxies[0].sersic.centre[0] == 0.01
        assert instance.galaxies[0].sis.centre[0] == 0.1
        assert instance.galaxies[0].sis.centre[1] == 0.2
        assert instance.galaxies[0].sis.einstein_radius == 10.0
        assert instance.galaxies[0].redshift == 0.4
        assert instance.galaxies[1].sis.centre[0] == 0.6
        assert instance.galaxies[1].sis.centre[1] == 0.5
        assert instance.galaxies[1].sis.einstein_radius == 0.7
        assert instance.galaxies[1].redshift == 0.8

    def test__phase_can_receive_hyper_image_and_noise_maps(self):
        phase_7x7 = al.PhaseImaging(
            galaxies=dict(
                lens=al.GalaxyModel(redshift=al.Redshift),
                lens1=al.GalaxyModel(redshift=al.Redshift),
            ),
            hyper_image_sky=al.HyperImageSky,
            hyper_background_noise=al.HyperBackgroundNoise,
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        instance = phase_7x7.optimizer.variable.instance_from_physical_vector(
            [0.1, 0.2, 0.3, 0.4]
        )

        assert instance.galaxies[0].redshift == 0.1
        assert instance.galaxies[1].redshift == 0.2
        assert instance.hyper_image_sky.sky_scale == 0.3
        assert instance.hyper_background_noise.noise_scale == 0.4

    def test__extended_with_hyper_and_pixelizations(self, phase_7x7):
        phase_extended = phase_7x7.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=False
        )
        assert phase_extended == phase_7x7

        phase_extended = phase_7x7.extend_with_multiple_hyper_phases(inversion=True)
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase

        phase_extended = phase_7x7.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=False
        )
        assert type(phase_extended.hyper_phases[0]) == al.HyperGalaxyPhase

        phase_extended = phase_7x7.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase

        phase_extended = phase_7x7.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.HyperGalaxyPhase
        assert type(phase_extended.hyper_phases[1]) == al.InversionPhase


class TestResult(object):
    def test__results_of_phase_are_available_as_properties(
        self, ccd_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_7x7 = al.PhaseImaging(
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

        result = phase_7x7.run(data=ccd_data_7x7)

        assert isinstance(result, al.AbstractPhase.Result)

    def test__results_of_phase_include_mask__available_as_property(
        self, ccd_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_7x7 = al.PhaseImaging(
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

        result = phase_7x7.run(data=ccd_data_7x7)

        mask = mask_function_7x7(image=ccd_data_7x7.image)

        assert (result.mask_2d == mask).all()

    def test__results_of_phase_include_positions__available_as_property(
        self, ccd_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_7x7 = al.PhaseImaging(
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

        result = phase_7x7.run(data=ccd_data_7x7)

        assert result.positions == None

        phase_7x7 = al.PhaseImaging(
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

        result = phase_7x7.run(data=ccd_data_7x7, positions=[[[1.0, 1.0]]])

        assert (result.positions[0] == np.array([1.0, 1.0])).all()

    def test__results_of_phase_include_pixelization__available_as_property(
        self, ccd_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_7x7 = al.PhaseImaging(
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

        result = phase_7x7.run(data=ccd_data_7x7)

        assert isinstance(result.pixelization, al.pixelizations.VoronoiMagnification)
        assert result.pixelization.shape == (2, 3)

        phase_7x7 = al.PhaseImaging(
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

        phase_7x7.galaxies.source.binned_hyper_galaxy_image_1d = np.ones(9)

        result = phase_7x7.run(data=ccd_data_7x7)

        assert isinstance(result.pixelization, al.pixelizations.VoronoiBrightnessImage)
        assert result.pixelization.pixels == 6

    def test__results_of_phase_include_pixelization_grid__available_as_property(
        self, ccd_data_7x7, mask_function_7x7
    ):
        clean_images()

        phase_7x7 = al.PhaseImaging(
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

        result = phase_7x7.run(data=ccd_data_7x7)

        assert result.most_likely_pixelization_grids_of_planes == None

        phase_7x7 = al.PhaseImaging(
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

        phase_7x7.galaxies.source.binned_hyper_galaxy_image_1d = np.ones(9)

        result = phase_7x7.run(data=ccd_data_7x7)

        assert result.most_likely_pixelization_grids_of_planes.shape == (6, 2)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, ccd_data_7x7, mask_function_7x7
    ):
        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.light_profiles.EllipticalSersic(intensity=0.1)
        )

        phase_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[lens_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        instance = phase_7x7.variable.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase_7x7.mask_function(image=ccd_data_7x7.image)
        lens_data = al.LensData(ccd_data=ccd_data_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = al.LensProfileFit(lens_data=lens_data, tracer=tracer)

        assert fit.likelihood == fit_figure_of_merit

    def test__fit_figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, ccd_data_7x7, mask_function_7x7
    ):
        hyper_image_sky = al.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = al.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = al.Galaxy(
            redshift=0.5, light=al.light_profiles.EllipticalSersic(intensity=0.1)
        )

        phase_7x7 = al.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[lens_galaxy],
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        instance = phase_7x7.variable.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase_7x7.mask_function(image=ccd_data_7x7.image)
        lens_data = al.LensData(ccd_data=ccd_data_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = al.LensProfileFit(
            lens_data=lens_data,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.likelihood == fit_figure_of_merit


class TestPhasePickle(object):

    # noinspection PyTypeChecker
    def test_assertion_failure(self, ccd_data_7x7, mask_function_7x7):
        def make_analysis(*args, **kwargs):
            return mock_pipeline.MockAnalysis(1, 1)

        phase_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    light=al.light_profiles.EllipticalLightProfile, redshift=1
                )
            ),
        )

        phase_7x7.make_analysis = make_analysis
        result = phase_7x7.run(
            data=ccd_data_7x7, results=None, mask=None, positions=None
        )
        assert result is not None

        phase_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    light=al.light_profiles.EllipticalLightProfile, redshift=1
                )
            ),
        )

        phase_7x7.make_analysis = make_analysis
        result = phase_7x7.run(
            data=ccd_data_7x7, results=None, mask=None, positions=None
        )
        assert result is not None

        class CustomPhase(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies.lens.light = al.light_profiles.EllipticalLightProfile()

        phase_7x7 = CustomPhase(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(
                    light=al.light_profiles.EllipticalLightProfile, redshift=1
                )
            ),
        )
        phase_7x7.make_analysis = make_analysis

        # with pytest.raises(af.exc.PipelineException):
        #     phase_7x7.run(instrument=ccd_data_7x7, results=None, mask=None, positions=None)
