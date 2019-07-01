from autolens.pipeline import tagging

from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

class TestPipelineNameTag:

    def test__pipeline_name_and_tag__mixture_of_values(self):

        pipeline_name = tagging.pipeline_name_from_name_and_settings(
            pipeline_name='pl', fix_lens_light=True)

        assert pipeline_name == 'pl_fix_lens_light'

        pipeline_tag = tagging.pipeline_name_from_name_and_settings(
            pipeline_name='pl', fix_lens_light=True, pixelization=pix.Rectangular, regularization=reg.Constant)

        assert pipeline_tag == 'pl_fix_lens_light_pix_rect_reg_const'

        pipeline_name = tagging.pipeline_name_from_name_and_settings(
            pipeline_name='pl2', fix_lens_light=True, align_bulge_disk_phi=True)

        assert pipeline_name == 'pl2_fix_lens_light_bd_align_phi'

    def test__pipeline_tag__mixture_of_values(self):

        pipeline_tag = tagging.pipeline_tag_from_pipeline_settings(fix_lens_light=True)

        assert pipeline_tag == '_fix_lens_light'

        pipeline_tag = tagging.pipeline_tag_from_pipeline_settings(fix_lens_light=True, pixelization=pix.Rectangular,
                                                                   regularization=reg.Constant)

        assert pipeline_tag == '_fix_lens_light_pix_rect_reg_const'

        pipeline_tag = tagging.pipeline_tag_from_pipeline_settings(fix_lens_light=True, align_bulge_disk_phi=True)

        assert pipeline_tag == '_fix_lens_light_bd_align_phi'


class TestPipelineTaggers:

    def test__fix_lens_light_tagger(self):
        tag = tagging.fix_lens_light_tag_from_fix_lens_light(fix_lens_light=False)
        assert tag == ''
        tag = tagging.fix_lens_light_tag_from_fix_lens_light(fix_lens_light=True)
        assert tag == '_fix_lens_light'

    def test__pixelization_tagger(self):

        tag = tagging.pixelization_tag_from_pixelization(pixelization=None)
        assert tag == ''
        tag = tagging.pixelization_tag_from_pixelization(pixelization=pix.Rectangular)
        assert tag == '_pix_rect'
        tag = tagging.pixelization_tag_from_pixelization(pixelization=pix.VoronoiBrightnessImage)
        assert tag == '_pix_voro_image'

    def test__regularization_tagger(self):

        tag = tagging.regularization_tag_from_regularization(regularization=None)
        assert tag == ''
        tag = tagging.regularization_tag_from_regularization(regularization=reg.Constant)
        assert tag == '_reg_const'
        tag = tagging.regularization_tag_from_regularization(regularization=reg.AdaptiveBrightness)
        assert tag == '_reg_adapt_bright'

    def test__align_bulge_disk_taggers(self):
        tag = tagging.align_bulge_disk_centre_tag_from_align_bulge_disk_centre(align_bulge_disk_centre=False)
        assert tag == ''
        tag = tagging.align_bulge_disk_centre_tag_from_align_bulge_disk_centre(align_bulge_disk_centre=True)
        assert tag == '_bd_align_centre'

        tag = tagging.align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(
            align_bulge_disk_axis_ratio=False)
        assert tag == ''
        tag = tagging.align_bulge_disk_axis_ratio_tag_from_align_bulge_disk_axis_ratio(align_bulge_disk_axis_ratio=True)
        assert tag == '_bd_align_axis_ratio'

        tag = tagging.align_bulge_disk_phi_tag_from_align_bulge_disk_phi(align_bulge_disk_phi=False)
        assert tag == ''
        tag = tagging.align_bulge_disk_phi_tag_from_align_bulge_disk_phi(align_bulge_disk_phi=True)
        assert tag == '_bd_align_phi'

    def test__bulge_disk_tag(self):
        tag = tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=False, align_bulge_disk_axis_ratio=False, align_bulge_disk_phi=False)
        assert tag == ''

        tag = tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True, align_bulge_disk_axis_ratio=False, align_bulge_disk_phi=False)
        assert tag == '_bd_align_centre'

        tag = tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True, align_bulge_disk_axis_ratio=False, align_bulge_disk_phi=True)
        assert tag == '_bd_align_centre_bd_align_phi'

        tag = tagging.bulge_disk_tag_from_align_bulge_disks(
            align_bulge_disk_centre=True, align_bulge_disk_axis_ratio=True, align_bulge_disk_phi=True)
        assert tag == '_bd_align_centre_bd_align_axis_ratio_bd_align_phi'


class TestPhaseTag:

    def test__mixture_of_values(self):

        phase_tag = tagging.phase_tag_from_phase_settings(sub_grid_size=2,
                                                          bin_up_factor=None,
                                                          image_psf_shape=None,
                                                          inversion_psf_shape=None,
                                                          inner_mask_radii=0.3,
                                                          positions_threshold=2.0,
                                                          interp_pixel_scale=None,
                                                          cluster_pixel_scale=None)


        assert phase_tag == '_sub_2_pos_2.00_inner_mask_0.30'

        phase_tag = tagging.phase_tag_from_phase_settings(sub_grid_size=1,
                                                          bin_up_factor=3,
                                                          image_psf_shape=(2, 2),
                                                          inversion_psf_shape=(3,3),
                                                          inner_mask_radii=None,
                                                          positions_threshold=None,
                                                          interp_pixel_scale=0.2,
                                                          cluster_pixel_scale=0.3)

        assert phase_tag == '_sub_1_bin_up_3_image_psf_2x2_inv_psf_3x3_interp_0.200_cluster_0.300'


class TestPhaseTaggers:

    def test__positions_threshold_tagger(self):

        tag = tagging.positions_threshold_tag_from_positions_threshold(positions_threshold=None)
        assert tag == ''
        tag = tagging.positions_threshold_tag_from_positions_threshold(positions_threshold=1.0)
        assert tag == '_pos_1.00'
        tag = tagging.positions_threshold_tag_from_positions_threshold(positions_threshold=2.56)
        assert tag == '_pos_2.56'

    def test__inner_circular_mask_radii_tagger(self):

        tag = tagging.inner_mask_radii_tag_from_inner_circular_mask_radii(inner_mask_radii=None)
        assert tag == ''
        tag = tagging.inner_mask_radii_tag_from_inner_circular_mask_radii(inner_mask_radii=0.2)
        print(tag)
        assert tag == '_inner_mask_0.20'
        tag = tagging.inner_mask_radii_tag_from_inner_circular_mask_radii(inner_mask_radii=3)
        assert tag == '_inner_mask_3.00'

    def test__sub_grid_size_tagger(self):

        tag = tagging.sub_grid_size_tag_from_sub_grid_size(sub_grid_size=1)
        assert tag == '_sub_1'
        tag = tagging.sub_grid_size_tag_from_sub_grid_size(sub_grid_size=2)
        assert tag == '_sub_2'
        tag = tagging.sub_grid_size_tag_from_sub_grid_size(sub_grid_size=4)
        assert tag == '_sub_4'

    def test__bin_up_factor_tagger(self):

        tag = tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=None)
        assert tag == ''
        tag = tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=1)
        assert tag == ''
        tag = tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=2)
        assert tag == '_bin_up_2'
        tag = tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=3)
        assert tag == '_bin_up_3'

    def test__image_psf_shape_tagger(self):

        tag = tagging.image_psf_shape_tag_from_image_psf_shape(image_psf_shape=None)
        assert tag == ''
        tag = tagging.image_psf_shape_tag_from_image_psf_shape(image_psf_shape=(2,2))
        assert tag == '_image_psf_2x2'
        tag = tagging.image_psf_shape_tag_from_image_psf_shape(image_psf_shape=(3,4))
        assert tag == '_image_psf_3x4'

    def test__inversion_psf_shape_tagger(self):

        tag = tagging.inversion_psf_shape_tag_from_inversion_psf_shape(inversion_psf_shape=None)
        assert tag == ''
        tag = tagging.inversion_psf_shape_tag_from_inversion_psf_shape(inversion_psf_shape=(2,2))
        assert tag == '_inv_psf_2x2'
        tag = tagging.inversion_psf_shape_tag_from_inversion_psf_shape(inversion_psf_shape=(3,4))
        assert tag == '_inv_psf_3x4'

    def test__interp_pixel_scale_tagger(self):

        tag = tagging.interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale=None)
        assert tag == ''
        tag = tagging.interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale=0.5)
        assert tag == '_interp_0.500'
        tag = tagging.interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale=0.25)
        assert tag == '_interp_0.250'
        tag = tagging.interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale=0.234)
        assert tag == '_interp_0.234'
        
    def test__cluster_pixel_scale_tagger(self):

        tag = tagging.cluster_pixel_scale_tag_from_cluster_pixel_scale(cluster_pixel_scale=None)
        assert tag == ''
        tag = tagging.cluster_pixel_scale_tag_from_cluster_pixel_scale(cluster_pixel_scale=0.5)
        assert tag == '_cluster_0.500'
        tag = tagging.cluster_pixel_scale_tag_from_cluster_pixel_scale(cluster_pixel_scale=0.25)
        assert tag == '_cluster_0.250'
        tag = tagging.cluster_pixel_scale_tag_from_cluster_pixel_scale(cluster_pixel_scale=0.234)
        assert tag == '_cluster_0.234'