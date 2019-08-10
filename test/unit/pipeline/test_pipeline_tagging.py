from autolens.pipeline import pipeline_tagging

from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg


class TestPipelineNameTag:
    def test__pipeline_tag__mixture_of_values(self):

        pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
            hyper_galaxies=True, hyper_image_sky=True, hyper_background_noise=True
        )

        assert pipeline_tag == "pipeline_tag__hyper_galaxies_bg_sky_bg_noise"

        pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
            fix_lens_light=True
        )

        assert pipeline_tag == "pipeline_tag__fix_lens_light"

        pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
            fix_lens_light=True,
            pixelization=pix.Rectangular,
            regularization=reg.Constant,
        )

        assert pipeline_tag == "pipeline_tag__fix_lens_light__pix_rect__reg_const"

        pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
            fix_lens_light=True, align_bulge_disk_phi=True
        )

        assert pipeline_tag == "pipeline_tag__fix_lens_light__bd_align_phi"

        pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
            align_light_dark_centre=True,
            align_bulge_dark_centre=True,
            disk_as_sersic=True,
        )

        assert (
            pipeline_tag
            == "pipeline_tag__disk_sersic__light_dark_align_centre__bulge_dark_align_centre"
        )


class TestHyperPipelineTaggers:
    def test__hyper_galaxies_tagger(self):
        tag = pipeline_tagging.hyper_galaxies_tag_from_hyper_galaxies(
            hyper_galaxies=False
        )
        assert tag == ""
        tag = pipeline_tagging.hyper_galaxies_tag_from_hyper_galaxies(
            hyper_galaxies=True
        )
        assert tag == "_galaxies"

    def test__hyper_image_sky_tagger(self):
        tag = pipeline_tagging.hyper_image_sky_tag_from_hyper_image_sky(
            hyper_image_sky=False
        )
        assert tag == ""
        tag = pipeline_tagging.hyper_image_sky_tag_from_hyper_image_sky(
            hyper_image_sky=True
        )
        assert tag == "_bg_sky"

    def test__hyper_background_noise_tagger(self):
        tag = pipeline_tagging.hyper_background_noise_tag_from_hyper_background_noise(
            hyper_background_noise=False
        )
        assert tag == ""
        tag = pipeline_tagging.hyper_background_noise_tag_from_hyper_background_noise(
            hyper_background_noise=True
        )
        assert tag == "_bg_noise"

    def test__tag_from_hyper_settings(self):

        tag = pipeline_tagging.hyper_tag_from_hyper_settings(
            hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
        )

        assert tag == ""

        tag = pipeline_tagging.hyper_tag_from_hyper_settings(
            hyper_galaxies=True, hyper_image_sky=False, hyper_background_noise=False
        )

        assert tag == "__hyper_galaxies"

        tag = pipeline_tagging.hyper_tag_from_hyper_settings(
            hyper_galaxies=False, hyper_image_sky=True, hyper_background_noise=True
        )

        assert tag == "__hyper_bg_sky_bg_noise"


class TestPipelineTaggers:
    def test__include_shear_tagger(self):
        tag = pipeline_tagging.include_shear_tag_from_include_shear(include_shear=False)
        assert tag == ""
        tag = pipeline_tagging.include_shear_tag_from_include_shear(include_shear=True)
        assert tag == "__with_shear"

    def test__fix_lens_light_tagger(self):
        tag = pipeline_tagging.fix_lens_light_tag_from_fix_lens_light(
            fix_lens_light=False
        )
        assert tag == ""
        tag = pipeline_tagging.fix_lens_light_tag_from_fix_lens_light(
            fix_lens_light=True
        )
        assert tag == "__fix_lens_light"

    def test__pixelization_tagger(self):

        tag = pipeline_tagging.pixelization_tag_from_pixelization(pixelization=None)
        assert tag == ""
        tag = pipeline_tagging.pixelization_tag_from_pixelization(
            pixelization=pix.Rectangular
        )
        assert tag == "__pix_rect"
        tag = pipeline_tagging.pixelization_tag_from_pixelization(
            pixelization=pix.VoronoiBrightnessImage
        )
        assert tag == "__pix_voro_image"

    def test__regularization_tagger(self):

        tag = pipeline_tagging.regularization_tag_from_regularization(
            regularization=None
        )
        assert tag == ""
        tag = pipeline_tagging.regularization_tag_from_regularization(
            regularization=reg.Constant
        )
        assert tag == "__reg_const"
        tag = pipeline_tagging.regularization_tag_from_regularization(
            regularization=reg.AdaptiveBrightness
        )
        assert tag == "__reg_adapt_bright"

    def test__align_bulge_disk_taggers(self):
        tag = pipeline_tagging.align_bulge_disk_centre_tag_from_align_bulge_disk_centre(
            align_bulge_disk_centre=False
        )
        assert tag == ""
        tag = pipeline_tagging.align_bulge_disk_centre_tag_from_align_bulge_disk_centre(
            align_bulge_disk_centre=True
        )
        assert tag == "__bd_align_centre"

        tag = pipeline_tagging.align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
            align_bulge_disk_axis_ratio=False
        )
        assert tag == ""
        tag = pipeline_tagging.align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
            align_bulge_disk_axis_ratio=True
        )
        assert tag == "__bd_align_axis_ratio"

        tag = pipeline_tagging.align_bulge_disk_phi_tag_from_align_bulge_disk_phi(
            align_bulge_disk_phi=False
        )
        assert tag == ""
        tag = pipeline_tagging.align_bulge_disk_phi_tag_from_align_bulge_disk_phi(
            align_bulge_disk_phi=True
        )
        assert tag == "__bd_align_phi"

    def test__bulge_disk_tag(self):
        tag = pipeline_tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=False,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=False,
        )
        assert tag == ""

        tag = pipeline_tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=False,
        )
        assert tag == "__bd_align_centre"

        tag = pipeline_tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=True,
        )
        assert tag == "__bd_align_centre__bd_align_phi"

        tag = pipeline_tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=True,
            align_bulge_disk_phi=True,
        )
        assert tag == "__bd_align_centre__bd_align_axis_ratio__bd_align_phi"

    def test__disk_as_sersic_tagger(self):
        tag = pipeline_tagging.disk_as_sersic_tag_from_disk_as_sersic(
            disk_as_sersic=False
        )
        assert tag == ""
        tag = pipeline_tagging.disk_as_sersic_tag_from_disk_as_sersic(
            disk_as_sersic=True
        )
        assert tag == "__disk_sersic"

    def test__align_light_dark_tagger(self):
        tag = pipeline_tagging.align_light_dark_centre_tag_from_align_light_dark_centre(
            align_light_dark_centre=False
        )
        assert tag == ""
        tag = pipeline_tagging.align_light_dark_centre_tag_from_align_light_dark_centre(
            align_light_dark_centre=True
        )
        assert tag == "__light_dark_align_centre"

    def test__align_bulge_dark_tagger(self):

        tag = pipeline_tagging.align_bulge_dark_centre_tag_from_align_bulge_dark_centre(
            align_bulge_dark_centre=False
        )
        assert tag == ""
        tag = pipeline_tagging.align_bulge_dark_centre_tag_from_align_bulge_dark_centre(
            align_bulge_dark_centre=True
        )
        assert tag == "__bulge_dark_align_centre"
