from autolens.pipeline import pipeline_tagging

from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg


class TestPipelineNameTag:
    def test__pipeline_name_and_tag__mixture_of_values(self):

        pipeline_name = pipeline_tagging.pipeline_name_from_name_and_settings(
            pipeline_name="pl", fix_lens_light=True
        )

        assert pipeline_name == "pl_fix_lens_light"

        pipeline_tag = pipeline_tagging.pipeline_name_from_name_and_settings(
            pipeline_name="pl",
            include_shear=True,
            fix_lens_light=True,
            pixelization=pix.Rectangular,
            regularization=reg.Constant,
        )

        assert pipeline_tag == "pl_with_shear_fix_lens_light_pix_rect_reg_const"

        pipeline_name = pipeline_tagging.pipeline_name_from_name_and_settings(
            pipeline_name="pl2", fix_lens_light=True, align_bulge_disk_phi=True
        )

        assert pipeline_name == "pl2_fix_lens_light_bd_align_phi"

    def test__pipeline_tag__mixture_of_values(self):

        pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
            fix_lens_light=True
        )

        assert pipeline_tag == "_fix_lens_light"

        pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
            fix_lens_light=True,
            pixelization=pix.Rectangular,
            regularization=reg.Constant,
        )

        assert pipeline_tag == "_fix_lens_light_pix_rect_reg_const"

        pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
            fix_lens_light=True, align_bulge_disk_phi=True
        )

        assert pipeline_tag == "_fix_lens_light_bd_align_phi"


class TestPipelineTaggers:
    def test__fix_lens_light_tagger(self):
        tag = pipeline_tagging.fix_lens_light_tag_from_fix_lens_light(
            fix_lens_light=False
        )
        assert tag == ""
        tag = pipeline_tagging.fix_lens_light_tag_from_fix_lens_light(
            fix_lens_light=True
        )
        assert tag == "_fix_lens_light"

    def test__pixelization_tagger(self):

        tag = pipeline_tagging.pixelization_tag_from_pixelization(pixelization=None)
        assert tag == ""
        tag = pipeline_tagging.pixelization_tag_from_pixelization(
            pixelization=pix.Rectangular
        )
        assert tag == "_pix_rect"
        tag = pipeline_tagging.pixelization_tag_from_pixelization(
            pixelization=pix.VoronoiBrightnessImage
        )
        assert tag == "_pix_voro_image"

    def test__regularization_tagger(self):

        tag = pipeline_tagging.regularization_tag_from_regularization(
            regularization=None
        )
        assert tag == ""
        tag = pipeline_tagging.regularization_tag_from_regularization(
            regularization=reg.Constant
        )
        assert tag == "_reg_const"
        tag = pipeline_tagging.regularization_tag_from_regularization(
            regularization=reg.AdaptiveBrightness
        )
        assert tag == "_reg_adapt_bright"

    def test__align_bulge_disk_taggers(self):
        tag = pipeline_tagging.align_bulge_disk_centre_tag_from_align_bulge_disk_centre(
            align_bulge_disk_centre=False
        )
        assert tag == ""
        tag = pipeline_tagging.align_bulge_disk_centre_tag_from_align_bulge_disk_centre(
            align_bulge_disk_centre=True
        )
        assert tag == "_bd_align_centre"

        tag = pipeline_tagging.align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
            align_bulge_disk_axis_ratio=False
        )
        assert tag == ""
        tag = pipeline_tagging.align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
            align_bulge_disk_axis_ratio=True
        )
        assert tag == "_bd_align_axis_ratio"

        tag = pipeline_tagging.align_bulge_disk_phi_tag_from_align_bulge_disk_phi(
            align_bulge_disk_phi=False
        )
        assert tag == ""
        tag = pipeline_tagging.align_bulge_disk_phi_tag_from_align_bulge_disk_phi(
            align_bulge_disk_phi=True
        )
        assert tag == "_bd_align_phi"

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
        assert tag == "_bd_align_centre"

        tag = pipeline_tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=True,
        )
        assert tag == "_bd_align_centre_bd_align_phi"

        tag = pipeline_tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=True,
            align_bulge_disk_phi=True,
        )
        assert tag == "_bd_align_centre_bd_align_axis_ratio_bd_align_phi"
