from autolens.pipeline import tagging

class TestTaggers:

    def test__positions_threshold_tagger(self):

        tag = tagging.positions_threshold_tag_from_positions_threshold(positions_threshold=None)
        assert tag == ''
        tag = tagging.positions_threshold_tag_from_positions_threshold(positions_threshold=1.0)
        assert tag == '_positions_threshold_1.00'
        tag = tagging.positions_threshold_tag_from_positions_threshold(positions_threshold=2.56)
        assert tag == '_positions_threshold_2.56'

    def test__inner_circular_mask_radii_tagger(self):

        tag = tagging.inner_circular_mask_radii_tag_from_inner_circular_mask_radii(inner_circular_mask_radii=None)
        assert tag == ''
        tag = tagging.inner_circular_mask_radii_tag_from_inner_circular_mask_radii(inner_circular_mask_radii=0.2)
        print(tag)
        assert tag == '_inner_circular_mask_radii_0.20'
        tag = tagging.inner_circular_mask_radii_tag_from_inner_circular_mask_radii(inner_circular_mask_radii=3)
        assert tag == '_inner_circular_mask_radii_3.00'

    def test__sub_grid_size_tagger(self):

        tag = tagging.sub_grid_size_tag_from_sub_grid_size(sub_grid_size=1)
        assert tag == '_sub_grid_size_1'
        tag = tagging.sub_grid_size_tag_from_sub_grid_size(sub_grid_size=2)
        assert tag == '_sub_grid_size_2'
        tag = tagging.sub_grid_size_tag_from_sub_grid_size(sub_grid_size=4)
        assert tag == '_sub_grid_size_4'

    def test__bin_up_factor_tagger(self):

        tag = tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=1)
        assert tag == ''
        tag = tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=2)
        assert tag == '_bin_up_factor_2'
        tag = tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=3)
        assert tag == '_bin_up_factor_3'

    def test__image_psf_shape_tagger(self):

        tag = tagging.image_psf_shape_tag_from_image_psf_shape(image_psf_shape=None)
        assert tag == ''
        tag = tagging.image_psf_shape_tag_from_image_psf_shape(image_psf_shape=(2,2))
        print(tag.strip())
        assert tag == '_image_psf_shape_2x2'
        tag = tagging.image_psf_shape_tag_from_image_psf_shape(image_psf_shape=(3,4))
        assert tag == '_image_psf_shape_3x4'

    def test__inversion_psf_shape_tagger(self):

        tag = tagging.inversion_psf_shape_tag_from_inversion_psf_shape(inversion_psf_shape=None)
        assert tag == ''
        tag = tagging.inversion_psf_shape_tag_from_inversion_psf_shape(inversion_psf_shape=(2,2))
        print(tag.strip())
        assert tag == '_inversion_psf_shape_2x2'
        tag = tagging.inversion_psf_shape_tag_from_inversion_psf_shape(inversion_psf_shape=(3,4))
        assert tag == '_inversion_psf_shape_3x4'

    def test__interp_pixel_scale_tagger(self):

        tag = tagging.interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale=None)
        assert tag == ''
        tag = tagging.interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale=0.5)
        assert tag == '_interp_pixel_scale_0.500'
        tag = tagging.interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale=0.25)
        assert tag == '_interp_pixel_scale_0.250'
        tag = tagging.interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale=0.234)
        assert tag == '_interp_pixel_scale_0.234'