import os
from os import path

import numpy as np
import pytest
from astropy import cosmology as cosmo

import autofit as af
from autolens import exc
from autolens.data.array import mask as msk
from autolens.lens import ray_tracing
from autolens.lens import lens_data as ld
from autolens.lens import lens_fit
from autolens.model.hyper import hyper_data as hd
from autolens.model.galaxy import galaxy as g, galaxy_model as gm
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.pipeline.phase import phase
from autolens.pipeline.phase import phase_imaging
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

        phase_7x7.galaxies = [g.Galaxy(redshift=0.5)]
        assert phase_7x7.optimizer.variable.galaxies == [g.Galaxy(redshift=0.5)]

    def test_set_variables(self, phase_7x7):

        phase_7x7.galaxies = [gm.GalaxyModel(redshift=0.5)]
        assert phase_7x7.optimizer.variable.galaxies == [gm.GalaxyModel(redshift=0.5)]

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

        mask_input = msk.Mask.circular(
            shape=ccd_data_7x7.shape, pixel_scale=1, radius_arcsec=1.5
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7, mask=mask_input)

        assert (analysis.lens_data.mask_2d == mask_input).all()

        # If a mask function is suppled, we should use this mask, regardless of whether an input mask is supplied.

        def mask_function(image):
            return msk.Mask.circular(
                shape=image.shape, pixel_scale=1, radius_arcsec=0.3
            )

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

    def test_make_analysis__mask_input_uses_mask__inner_mask_radius_included_which_masks_centre(
        self, phase_7x7, ccd_data_7x7
    ):

        # If an input mask is supplied and there is no mask function, we use mask input.

        phase_7x7.mask_function = None
        phase_7x7.inner_mask_radii = 0.5

        mask_input = msk.Mask.circular(
            shape=ccd_data_7x7.shape, pixel_scale=1, radius_arcsec=1.5
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7, mask=mask_input)

        # The inner circulaar mask radii of 0.5" masks only the central pixel of the mask

        mask_input[3, 3] = True

        assert (analysis.lens_data.mask_2d == mask_input).all()

        # If a mask function is supplied, we should use this mask, regardless of whether an input mask is supplied.

        def mask_function(image):
            return msk.Mask.circular(
                shape=image.shape, pixel_scale=1, radius_arcsec=1.4
            )

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

    def test_make_analysis__positions_are_input__are_used_in_analysis(
        self, phase_7x7, ccd_data_7x7
    ):
        # If position threshold is input (not None) and positions are input, make the positions part of the lens data.

        phase_7x7.positions_threshold = 0.2

        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7, positions=[[[1.0, 1.0], [2.0, 2.0]]]
        )

        assert (analysis.lens_data.positions[0][0] == np.array([1.0, 1.0])).all()
        assert (analysis.lens_data.positions[0][1] == np.array([2.0, 2.0])).all()

        # If position threshold is input (not None) and but no positions are supplied, raise an error

        with pytest.raises(exc.PhaseException):
            phase_7x7.make_analysis(data=ccd_data_7x7, positions=None)
            phase_7x7.make_analysis(data=ccd_data_7x7)

        # If positions threshold is None, positions should always be None.

        phase_7x7.positions_threshold = None
        analysis = phase_7x7.make_analysis(
            data=ccd_data_7x7, positions=[[[1.0, 1.0], [2.0, 2.0]]]
        )

        assert analysis.lens_data.positions is None

    def test_make_analysis__inversion_resolution_error_raised_if_above_inversion_pixel_limit(
        self, phase_7x7, ccd_data_7x7, mask_function_7x7
    ):

        phase_7x7 = phase_imaging.PhaseImaging(
            galaxies=dict(
                source=g.Galaxy(
                    redshift=0.5,
                    pixelization=pix.Rectangular(shape=(3, 3)),
                    regularization=reg.Constant(),
                )
            ),
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        instance = phase_7x7.variable.instance_from_unit_vector([])

        analysis.check_inversion_pixels_are_below_limit(instance=instance)

        phase_7x7 = phase_imaging.PhaseImaging(
            galaxies=dict(
                source=g.Galaxy(
                    redshift=0.5,
                    pixelization=pix.Rectangular(shape=(4, 4)),
                    regularization=reg.Constant(),
                )
            ),
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        instance = phase_7x7.variable.instance_from_unit_vector([])

        with pytest.raises(exc.PixelizationException):
            analysis.check_inversion_pixels_are_below_limit(instance=instance)
            analysis.fit(instance=instance)

        phase_7x7 = phase_imaging.PhaseImaging(
            galaxies=dict(
                source=g.Galaxy(
                    redshift=0.5,
                    pixelization=pix.Rectangular(shape=(3, 3)),
                    regularization=reg.Constant(),
                )
            ),
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        instance = phase_7x7.variable.instance_from_unit_vector([])

        analysis.check_inversion_pixels_are_below_limit(instance=instance)

        phase_7x7 = phase_imaging.PhaseImaging(
            galaxies=dict(
                source=g.Galaxy(
                    redshift=0.5,
                    pixelization=pix.Rectangular(shape=(4, 4)),
                    regularization=reg.Constant(),
                )
            ),
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        instance = phase_7x7.variable.instance_from_unit_vector([])

        with pytest.raises(exc.PixelizationException):
            analysis.check_inversion_pixels_are_below_limit(instance=instance)
            analysis.fit(instance=instance)

    def test_make_analysis__interp_pixel_scale_is_input__interp_grid_used_in_analysis(
        self, phase_7x7, ccd_data_7x7
    ):
        # If use positions is true and positions are input, make the positions part of the lens data.

        phase_7x7.interp_pixel_scale = 0.1

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        assert analysis.lens_data.interp_pixel_scale == 0.1
        assert hasattr(analysis.lens_data.grid_stack.regular, "interpolator")
        assert hasattr(analysis.lens_data.grid_stack.sub, "interpolator")
        assert hasattr(analysis.lens_data.grid_stack.blurring, "interpolator")

    def test_make_analysis__cluster_pixel_limit__is_input__used_in_analysis(
        self, phase_7x7, ccd_data_7x7
    ):

        phase_7x7.galaxies.lens = gm.GalaxyModel(
            redshift=0.5,
            pixelization=pix.VoronoiBrightnessImage,
            regularization=reg.Constant,
        )

        phase_7x7.cluster_pixel_limit = 5

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        assert analysis.lens_data.cluster_pixel_limit == 5

        phase_7x7.cluster_pixel_limit = 10

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
        image_psf_shape = phase_info.readline()
        pixelization_psf_shape = phase_info.readline()
        positions_threshold = phase_info.readline()
        cosmology = phase_info.readline()
        auto_link_priors = phase_info.readline()

        phase_info.close()

        assert optimizer == "Optimizer = MockNLO \n"
        assert sub_grid_size == "Sub-grid size = 2 \n"
        assert image_psf_shape == "Image PSF shape = None \n"
        assert pixelization_psf_shape == "Pixelization PSF shape = None \n"
        assert positions_threshold == "Positions Threshold = None \n"
        assert (
            cosmology
            == 'Cosmology = FlatLambdaCDM(name="Planck15", H0=67.7 km / (Mpc s), Om0=0.307, Tcmb0=2.725 K, '
            "Neff=3.05, m_nu=[0.   0.   0.06] eV, Ob0=0.0486) \n"
        )
        assert auto_link_priors == "Auto Link Priors = False \n"

    def test_fit(self, ccd_data_7x7, mask_function_7x7):

        clean_images()

        phase_7x7 = phase_imaging.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic),
                source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic),
            ),
            mask_function=mask_function_7x7,
            phase_name="test_phase_test_fit",
        )

        result = phase_7x7.run(data=ccd_data_7x7)
        assert isinstance(result.constant.galaxies[0], g.Galaxy)
        assert isinstance(result.constant.galaxies[0], g.Galaxy)

    def test_customize(
        self, mask_function_7x7, results_7x7, results_collection_7x7, ccd_data_7x7
    ):
        class MyPlanePhaseAnd(phase_imaging.PhaseImaging):
            def pass_priors(self, results):

                self.galaxies = results.last.constant.galaxies

        galaxy = g.Galaxy(redshift=0.5)
        galaxy_model = gm.GalaxyModel(redshift=0.5)

        setattr(results_7x7.constant, "galaxies", [galaxy])
        setattr(results_7x7.variable, "galaxies", [galaxy_model])

        phase_7x7 = MyPlanePhaseAnd(
            phase_name="test_phase",
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
        )

        phase_7x7.make_analysis(data=ccd_data_7x7, results=results_collection_7x7)
        phase_7x7.pass_priors(results_collection_7x7)

        assert phase_7x7.galaxies == [galaxy]

        class MyPlanePhaseAnd(phase_imaging.PhaseImaging):
            def pass_priors(self, results):

                self.galaxies = results.last.variable.galaxies

        galaxy = g.Galaxy(redshift=0.5)
        galaxy_model = gm.GalaxyModel(redshift=0.5)

        setattr(results_7x7.constant, "galaxies", [galaxy])
        setattr(results_7x7.variable, "galaxies", [galaxy_model])

        phase_7x7 = MyPlanePhaseAnd(
            phase_name="test_phase",
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
        )

        phase_7x7.make_analysis(data=ccd_data_7x7, results=results_collection_7x7)
        phase_7x7.pass_priors(results_collection_7x7)

        assert phase_7x7.galaxies == [galaxy_model]

    def test_default_mask_function(self, phase_7x7, ccd_data_7x7):
        lens_data = ld.LensData(
            ccd_data=ccd_data_7x7, mask=phase_7x7.mask_function(ccd_data_7x7.image)
        )

        assert len(lens_data.image_1d) == 9

    def test_duplication(self):

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5), source=gm.GalaxyModel(redshift=1.0)
            ),
        )

        phase_imaging.PhaseImaging(phase_name="test_phase")

        assert phase_7x7.galaxies is not None

    def test_modify_image(self, mask_function_7x7, ccd_data_7x7):
        class MyPhase(phase_imaging.PhaseImaging):
            def modify_image(self, image, results):

                assert ccd_data_7x7.image.shape == image.shape
                image = 20.0 * np.ones(shape=(5, 5))
                return image

        phase_7x7 = MyPhase(phase_name="phase_7x7", mask_function=mask_function_7x7)

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        assert (analysis.lens_data.unmasked_image == 20.0 * np.ones(shape=(5, 5))).all()
        assert (analysis.lens_data.image_1d == 20.0 * np.ones(shape=9)).all()

    def test__check_if_phase_uses_inversion(self, mask_function_7x7):

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(lens=gm.GalaxyModel(redshift=0.5)),
        )

        assert phase_7x7.uses_inversion is False

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(
                    redshift=0.5,
                    pixelization=pix.Rectangular,
                    regularization=reg.Constant,
                )
            ),
        )

        assert phase_7x7.uses_inversion is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5), source=gm.GalaxyModel(redshift=1.0)
            ),
        )

        assert phase_7x7.uses_inversion is False

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(
                    redshift=0.5,
                    pixelization=pix.Rectangular,
                    regularization=reg.Constant,
                ),
                source=gm.GalaxyModel(redshift=1.0),
            ),
        )

        assert phase_7x7.uses_inversion is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5),
                source=gm.GalaxyModel(
                    redshift=1.0,
                    pixelization=pix.Rectangular,
                    regularization=reg.Constant,
                ),
            ),
        )

        assert phase_7x7.uses_inversion is True

    def test__check_if_phase_uses_cluster_inversion(self, mask_function_7x7):

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5), source=gm.GalaxyModel(redshift=1.0)
            ),
        )

        assert phase_7x7.uses_cluster_inversion is False

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(
                    redshift=0.5,
                    pixelization=pix.Rectangular,
                    regularization=reg.Constant,
                ),
                source=gm.GalaxyModel(redshift=1.0),
            ),
        )

        assert phase_7x7.uses_cluster_inversion is False

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5),
                source=gm.GalaxyModel(
                    redshift=1.0,
                    pixelization=pix.VoronoiBrightnessImage,
                    regularization=reg.Constant,
                ),
            ),
        )

        assert phase_7x7.uses_cluster_inversion is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5), source=gm.GalaxyModel(redshift=1.0)
            ),
        )

        assert phase_7x7.uses_cluster_inversion is False

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(
                    redshift=0.5,
                    pixelization=pix.Rectangular,
                    regularization=reg.Constant,
                ),
                source=gm.GalaxyModel(redshift=1.0),
            ),
        )

        assert phase_7x7.uses_cluster_inversion is False

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5),
                source=gm.GalaxyModel(
                    redshift=1.0,
                    pixelization=pix.VoronoiBrightnessImage,
                    regularization=reg.Constant,
                ),
            ),
        )

        assert phase_7x7.uses_cluster_inversion is True

    def test__check_if_phase_uses_hyper_images(self, mask_function_7x7):

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(lens=gm.GalaxyModel(redshift=0.5)),
        )

        assert phase_7x7.uses_hyper_images is False

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(
                    redshift=0.5,
                    pixelization=pix.Rectangular,
                    regularization=reg.AdaptiveBrightness,
                )
            ),
        )

        assert phase_7x7.uses_hyper_images is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5, hyper_galaxy=g.HyperGalaxy)
            ),
        )

        assert phase_7x7.uses_hyper_images is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(
                    redshift=0.5,
                    pixelization=pix.Rectangular,
                    regularization=reg.Constant,
                ),
                source=gm.GalaxyModel(redshift=1.0),
            ),
        )

        assert phase_7x7.uses_hyper_images is False

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(
                    redshift=0.5,
                    pixelization=pix.Rectangular,
                    regularization=reg.AdaptiveBrightness,
                ),
                source=gm.GalaxyModel(redshift=1.0),
            ),
        )

        assert phase_7x7.uses_hyper_images is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5),
                source=gm.GalaxyModel(
                    redshift=1.0,
                    pixelization=pix.Rectangular,
                    regularization=reg.AdaptiveBrightness,
                ),
            ),
        )

        assert phase_7x7.uses_hyper_images is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5),
                source=gm.GalaxyModel(redshift=1.0, hyper_galaxy=g.HyperGalaxy),
            ),
        )

        assert phase_7x7.uses_hyper_images is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5), source=gm.GalaxyModel(redshift=1.0)
            ),
        )

        assert phase_7x7.uses_hyper_images is False

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5),
                source=gm.GalaxyModel(redshift=1.0, hyper_galaxy=g.HyperGalaxy),
            ),
        )

        assert phase_7x7.uses_hyper_images is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5, hyper_galaxy=g.HyperGalaxy),
                source=gm.GalaxyModel(redshift=1.0),
            ),
        )

        assert phase_7x7.uses_hyper_images is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=0.5),
                source=gm.GalaxyModel(
                    redshift=1.0,
                    pixelization=pix.Rectangular,
                    regularization=reg.AdaptiveBrightness,
                ),
            ),
        )

        assert phase_7x7.uses_hyper_images is True

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=gm.GalaxyModel(
                    redshift=0.5,
                    pixelization=pix.Rectangular,
                    regularization=reg.AdaptiveBrightness,
                ),
                source=gm.GalaxyModel(redshift=1.0),
            ),
        )

        assert phase_7x7.uses_hyper_images is True

    def test__use_border__determines_if_border_pixel_relocation_is_used(
        self, ccd_data_7x7, mask_function_7x7, lens_data_7x7
    ):

        lens_data_7x7.grid_stack.regular[4] = np.array([100.0, 100.0])

        # noinspection PyTypeChecker

        lens_galaxy = g.Galaxy(
            redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0)
        )
        source_galaxy = g.Galaxy(
            redshift=1.0,
            pixelization=pix.Rectangular(shape=(5, 5)),
            regularization=reg.Constant(coefficient=1.0),
        )

        tracer_no_border = ray_tracing.Tracer.from_galaxies_and_image_plane_grid_stack(
            galaxies=[lens_galaxy, source_galaxy],
            image_plane_grid_stack=lens_data_7x7.grid_stack,
            border=None,
        )

        mapper_no_border = tracer_no_border.mappers_of_planes[0]

        tracer_with_border = ray_tracing.Tracer.from_galaxies_and_image_plane_grid_stack(
            galaxies=[lens_galaxy, source_galaxy],
            image_plane_grid_stack=lens_data_7x7.grid_stack,
            border=lens_data_7x7.border,
        )

        mapper_with_border = tracer_with_border.mappers_of_planes[0]

        phase_7x7 = phase_imaging.PhaseImaging(
            galaxies=[lens_galaxy, source_galaxy],
            mask_function=mask_function_7x7,
            cosmology=cosmo.Planck15,
            phase_name="test_phase",
            use_inversion_border=True,
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        analysis.lens_data.grid_stack.regular[4] = np.array([100.0, 100.0])

        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        mapper = tracer.mappers_of_planes[0]

        assert (
            mapper_with_border.grid_stack.regular == mapper.grid_stack.regular
        ).all()

        phase_7x7 = phase_imaging.PhaseImaging(
            galaxies=[lens_galaxy, source_galaxy],
            mask_function=mask_function_7x7,
            cosmology=cosmo.Planck15,
            phase_name="test_phase",
            use_inversion_border=False,
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        analysis.lens_data.grid_stack.regular[4] = np.array([100.0, 100.0])

        instance = phase_7x7.variable.instance_from_unit_vector([])
        tracer = analysis.tracer_for_instance(instance=instance)
        mapper = tracer.mappers_of_planes[0]

        assert (mapper_no_border.grid_stack.regular == mapper.grid_stack.regular).all()

    def test__inversion_and_cluster_pixel_limit_computed_via_input_of_max_inversion_pixel_limit_and_prior_config(
        self, mask_function_7x7
    ):

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="phase_7x7",
            mask_function=mask_function_7x7,
            inversion_pixel_limit=None,
        )

        assert phase_7x7.inversion_pixel_limit == 1500
        assert phase_7x7.cluster_pixel_limit == 1500

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="phase_7x7",
            mask_function=mask_function_7x7,
            inversion_pixel_limit=10,
        )

        assert phase_7x7.inversion_pixel_limit == 10
        assert phase_7x7.cluster_pixel_limit == 10

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="phase_7x7",
            mask_function=mask_function_7x7,
            inversion_pixel_limit=2000,
        )

        assert phase_7x7.inversion_pixel_limit == 2000
        assert phase_7x7.cluster_pixel_limit == 1500

    def test__adds_pixelization_grid_to_grid_stack_if_required(
        self, ccd_data_7x7, mask_function_7x7
    ):

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase", mask_function=mask_function_7x7
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        galaxy = g.Galaxy(redshift=0.5)

        grid_stack = analysis.add_grids_to_grid_stack(
            galaxies=[galaxy, galaxy], grid_stack=analysis.lens_data.grid_stack
        )

        assert (grid_stack.pixelization == np.array([[0.0, 0.0]])).all()

        galaxy_pix_which_doesnt_use_pix_grid = g.Galaxy(
            redshift=0.5, pixelization=pix.Rectangular(), regularization=reg.Constant()
        )

        grid_stack = analysis.add_grids_to_grid_stack(
            galaxies=[galaxy_pix_which_doesnt_use_pix_grid],
            grid_stack=analysis.lens_data.grid_stack,
        )

        assert (grid_stack.pixelization == np.array([[0.0, 0.0]])).all()

        galaxy_pix_which_uses_pix_grid = g.Galaxy(
            redshift=0.5,
            pixelization=pix.VoronoiMagnification(),
            regularization=reg.Constant(),
        )

        grid_stack = analysis.add_grids_to_grid_stack(
            galaxies=[galaxy_pix_which_uses_pix_grid],
            grid_stack=analysis.lens_data.grid_stack,
        )

        assert (
            grid_stack.pixelization
            == np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )
        ).all()

        galaxy_pix_which_uses_brightness = g.Galaxy(
            redshift=0.5,
            pixelization=pix.VoronoiBrightnessImage(pixels=9),
            regularization=reg.Constant(),
        )

        galaxy_pix_which_uses_brightness.hyper_galaxy_cluster_image_1d = np.array(
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        )

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                lens=gm.GalaxyModel(
                    redshift=0.5,
                    pixelization=pix.VoronoiBrightnessImage,
                    regularization=reg.Constant,
                )
            ),
            inversion_pixel_limit=5,
            mask_function=mask_function_7x7,
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        grid_stack = analysis.add_grids_to_grid_stack(
            galaxies=[galaxy_pix_which_uses_brightness],
            grid_stack=analysis.lens_data.grid_stack,
        )

        assert (
            grid_stack.pixelization
            == np.array(
                [
                    [0.0, 1.0],
                    [1.0, -1.0],
                    [-1.0, -1.0],
                    [-1.0, 1.0],
                    [0.0, -1.0],
                    [1.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                ]
            )
        ).all()

    def test__phase_with_no_inversion__convolver_mapping_matrix_of_lens_data_is_none(
        self, ccd_data_7x7, mask_function_7x7
    ):

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="test_phase",
            mask_function=mask_function_7x7,
            galaxies=dict(lens=gm.GalaxyModel(redshift=0.5)),
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        assert analysis.lens_data.convolver_mapping_matrix is None

    def test__lens_data_signal_to_noise_limit(
        self, ccd_data_7x7, mask_7x7_1_pix, mask_function_7x7_1_pix
    ):

        ccd_data_snr_limit = ccd_data_7x7.new_ccd_data_with_signal_to_noise_limit(
            signal_to_noise_limit=1.0
        )

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="phase_7x7",
            signal_to_noise_limit=1.0,
            mask_function=mask_function_7x7_1_pix,
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        assert (analysis.lens_data.unmasked_image == ccd_data_snr_limit.image).all()
        assert (
            analysis.lens_data.unmasked_noise_map == ccd_data_snr_limit.noise_map
        ).all()

        lens_data = ld.LensData(ccd_data=ccd_data_7x7, mask=mask_7x7_1_pix)

        ccd_data_snr_limit = ccd_data_7x7.new_ccd_data_with_signal_to_noise_limit(
            signal_to_noise_limit=0.1
        )

        phase_7x7 = phase_imaging.PhaseImaging(
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

        phase_7x7 = phase_imaging.PhaseImaging(
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

        lens_data = ld.LensData(ccd_data=ccd_data_7x7, mask=mask_7x7_1_pix)

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

        lens_galaxy = g.Galaxy(redshift=0.5)
        source_galaxy = g.Galaxy(redshift=0.5)

        phase_7x7 = phase_imaging.PhaseImaging(
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

        phase_7x7 = phase_imaging.PhaseImaging(
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

        galaxy_0 = g.Galaxy(redshift=0.1)
        galaxy_1 = g.Galaxy(redshift=0.2)
        galaxy_2 = g.Galaxy(redshift=0.3)

        phase_7x7 = phase_imaging.PhaseImaging(
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

        lens_galaxy = g.Galaxy(redshift=0.5, light=lp.EllipticalSersic(intensity=0.1))

        phase_7x7 = phase_imaging.PhaseImaging(
            galaxies=[lens_galaxy],
            mask_function=mask_function_7x7,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)

        instance = phase_7x7.variable.instance_from_unit_vector([])

        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase_7x7.mask_function(image=ccd_data_7x7.image)
        lens_data = ld.LensData(ccd_data=ccd_data_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = lens_fit.LensProfileFit(lens_data=lens_data, tracer=tracer)

        assert fit.likelihood == fit_figure_of_merit

    def test__phase_can_receive_list_of_galaxy_models(self):

        phase_7x7 = phase_imaging.PhaseImaging(
            galaxies=dict(
                lens=gm.GalaxyModel(
                    sersic=lp.EllipticalSersic,
                    sis=mp.SphericalIsothermal,
                    redshift=g.Redshift,
                ),
                lens1=gm.GalaxyModel(sis=mp.SphericalIsothermal, redshift=g.Redshift),
            ),
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        instance = phase_7x7.optimizer.variable.instance_from_physical_vector(
            [
                0.2,
                0.21,
                0.22,
                0.23,
                0.24,
                0.25,
                0.8,
                0.1,
                0.2,
                0.3,
                0.4,
                0.9,
                0.5,
                0.7,
                0.8,
            ]
        )

        assert instance.galaxies[0].sersic.centre[0] == 0.2
        assert instance.galaxies[0].sis.centre[0] == 0.1
        assert instance.galaxies[0].sis.centre[1] == 0.2
        assert instance.galaxies[0].sis.einstein_radius == 0.3
        assert instance.galaxies[0].redshift == 0.4
        assert instance.galaxies[1].sis.centre[0] == 0.9
        assert instance.galaxies[1].sis.centre[1] == 0.5
        assert instance.galaxies[1].sis.einstein_radius == 0.7
        assert instance.galaxies[1].redshift == 0.8

        class LensPlanePhase2(phase_imaging.PhaseImaging):
            # noinspection PyUnusedLocal
            def pass_models(self, results):
                self.galaxies[0].sis.einstein_radius = 10.0

        phase_7x7 = LensPlanePhase2(
            galaxies=dict(
                lens=gm.GalaxyModel(
                    sersic=lp.EllipticalSersic,
                    sis=mp.SphericalIsothermal,
                    redshift=g.Redshift,
                ),
                lens1=gm.GalaxyModel(sis=mp.SphericalIsothermal, redshift=g.Redshift),
            ),
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        # noinspection PyTypeChecker
        phase_7x7.pass_models(None)

        instance = phase_7x7.optimizer.variable.instance_from_physical_vector(
            [
                0.01,
                0.02,
                0.23,
                0.04,
                0.05,
                0.06,
                0.87,
                0.1,
                0.2,
                0.4,
                0.5,
                0.5,
                0.7,
                0.8,
            ]
        )

        assert instance.galaxies[0].sersic.centre[0] == 0.01
        assert instance.galaxies[0].sis.centre[0] == 0.1
        assert instance.galaxies[0].sis.centre[1] == 0.2
        assert instance.galaxies[0].sis.einstein_radius == 10.0
        assert instance.galaxies[0].redshift == 0.4
        assert instance.galaxies[1].sis.centre[0] == 0.5
        assert instance.galaxies[1].sis.centre[1] == 0.5
        assert instance.galaxies[1].sis.einstein_radius == 0.7
        assert instance.galaxies[1].redshift == 0.8

    def test__phase_can_receive_hyper_image_and_noise_maps(self):

        phase_7x7 = phase_imaging.PhaseImaging(
            galaxies=dict(
                lens=gm.GalaxyModel(redshift=g.Redshift),
                lens1=gm.GalaxyModel(redshift=g.Redshift),
            ),
            hyper_image_sky=hd.HyperImageSky,
            hyper_noise_background=hd.HyperNoiseBackground,
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        instance = phase_7x7.optimizer.variable.instance_from_physical_vector(
            [0.1, 0.2, 0.3, 0.4]
        )

        assert instance.galaxies[0].redshift == 0.1
        assert instance.galaxies[1].redshift == 0.2
        assert instance.hyper_image_sky.background_sky_scale == 0.3
        assert instance.hyper_noise_background.background_noise_scale == 0.4

    def test__extended_with_hyper_and_pixelizations(self, phase_7x7):

        from autolens.pipeline.phase import phase_extensions

        phase_extended = phase_7x7.extend_with_multiple_hyper_phases(inversion=True)
        assert type(phase_extended.hyper_phases[0]) == phase_extensions.InversionPhase

        phase_extended = phase_7x7.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=False
        )
        assert phase_extended.hyper_phases == []

        phase_extended = phase_7x7.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=False
        )
        assert type(phase_extended.hyper_phases[0]) == phase_extensions.HyperGalaxyPhase

        phase_extended = phase_7x7.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == phase_extensions.InversionPhase

        phase_extended = phase_7x7.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True
        )
        assert type(phase_extended.hyper_phases[1]) == phase_extensions.HyperGalaxyPhase
        assert type(phase_extended.hyper_phases[0]) == phase_extensions.InversionPhase


class TestResult(object):
    def test__results_of_phase_are_available_as_properties(
        self, ccd_data_7x7, mask_function_7x7
    ):

        clean_images()

        phase_7x7 = phase_imaging.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=[g.Galaxy(redshift=0.5, light=lp.EllipticalSersic(intensity=1.0))],
            phase_name="test_phase_2",
        )

        result = phase_7x7.run(data=ccd_data_7x7)

        assert isinstance(result, phase.AbstractPhase.Result)

    def test__results_of_phase_include_pixelization_grid__available_as_property(
        self, ccd_data_7x7, mask_function_7x7
    ):

        clean_images()

        phase_7x7 = phase_imaging.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=[g.Galaxy(redshift=0.5, light=lp.EllipticalSersic(intensity=1.0))],
            phase_name="test_phase_2",
        )

        result = phase_7x7.run(data=ccd_data_7x7)

        assert result.most_likely_image_plane_pixelization_grid == None

        phase_7x7 = phase_imaging.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            mask_function=mask_function_7x7,
            galaxies=dict(
                lens=g.Galaxy(redshift=0.5, light=lp.EllipticalSersic(intensity=1.0)),
                source=g.Galaxy(
                    redshift=1.0,
                    pixelization=pix.VoronoiBrightnessImage(pixels=6),
                    regularization=reg.Constant(),
                ),
            ),
            inversion_pixel_limit=6,
            phase_name="test_phase_2",
        )

        phase_7x7.galaxies.source.hyper_galaxy_cluster_image_1d = np.ones(9)

        result = phase_7x7.run(data=ccd_data_7x7)

        assert result.most_likely_image_plane_pixelization_grid.shape == (6, 2)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, ccd_data_7x7, mask_function_7x7
    ):

        lens_galaxy = g.Galaxy(redshift=0.5, light=lp.EllipticalSersic(intensity=0.1))

        phase_7x7 = phase_imaging.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[lens_galaxy],
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        instance = phase_7x7.variable.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase_7x7.mask_function(image=ccd_data_7x7.image)
        lens_data = ld.LensData(ccd_data=ccd_data_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = lens_fit.LensProfileFit(lens_data=lens_data, tracer=tracer)

        assert fit.likelihood == fit_figure_of_merit

    def test__fit_figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, ccd_data_7x7, mask_function_7x7
    ):

        hyper_image_sky = hd.HyperImageSky(background_sky_scale=1.0)
        hyper_noise_background = hd.HyperNoiseBackground(background_noise_scale=1.0)

        lens_galaxy = g.Galaxy(redshift=0.5, light=lp.EllipticalSersic(intensity=0.1))

        phase_7x7 = phase_imaging.PhaseImaging(
            mask_function=mask_function_7x7,
            galaxies=[lens_galaxy],
            hyper_image_sky=hyper_image_sky,
            hyper_noise_background=hyper_noise_background,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_7x7.make_analysis(data=ccd_data_7x7)
        instance = phase_7x7.variable.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase_7x7.mask_function(image=ccd_data_7x7.image)
        lens_data = ld.LensData(ccd_data=ccd_data_7x7, mask=mask)
        tracer = analysis.tracer_for_instance(instance=instance)
        fit = lens_fit.LensProfileFit(
            lens_data=lens_data,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_noise_background=hyper_noise_background,
        )

        assert fit.likelihood == fit_figure_of_merit


class TestPhasePickle(object):

    # noinspection PyTypeChecker
    def test_assertion_failure(self, ccd_data_7x7, mask_function_7x7):
        def make_analysis(*args, **kwargs):
            return mock_pipeline.MockAnalysis(1, 1)

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(lens=g.Galaxy(light=lp.EllipticalLightProfile, redshift=1)),
        )

        phase_7x7.make_analysis = make_analysis
        result = phase_7x7.run(
            data=ccd_data_7x7, results=None, mask=None, positions=None
        )
        assert result is not None

        phase_7x7 = phase_imaging.PhaseImaging(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(lens=g.Galaxy(light=lp.EllipticalLightProfile, redshift=1)),
        )

        phase_7x7.make_analysis = make_analysis
        result = phase_7x7.run(
            data=ccd_data_7x7, results=None, mask=None, positions=None
        )
        assert result is not None

        class CustomPhase(phase_imaging.PhaseImaging):
            def pass_priors(self, results):

                self.galaxies.lens.light = lp.EllipticalLightProfile()

        phase_7x7 = CustomPhase(
            phase_name="phase_name",
            mask_function=mask_function_7x7,
            optimizer_class=mock_pipeline.MockNLO,
            galaxies=dict(lens=g.Galaxy(light=lp.EllipticalLightProfile, redshift=1)),
        )
        phase_7x7.make_analysis = make_analysis

        # with pytest.raises(af.exc.PipelineException):
        #     phase_7x7.run(data=ccd_data_7x7, results=None, mask=None, positions=None)
