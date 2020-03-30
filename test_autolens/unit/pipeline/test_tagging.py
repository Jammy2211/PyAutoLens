import autolens as al


class TestPhaseTag:
    def test__mixture_of_values(self):

        phase_tag = al.tagging.phase_tag_from_phase_settings(
            sub_size=2,
            signal_to_noise_limit=2,
            bin_up_factor=None,
            psf_shape_2d=None,
            positions_threshold=2.0,
            pixel_scale_interpolation_grid=None,
        )

        assert phase_tag == "phase_tag__sub_2__snr_2__pos_2.00"

        phase_tag = al.tagging.phase_tag_from_phase_settings(
            sub_size=1,
            signal_to_noise_limit=None,
            bin_up_factor=3,
            psf_shape_2d=(2, 2),
            positions_threshold=None,
            pixel_scale_interpolation_grid=0.2,
        )

        assert phase_tag == "phase_tag__sub_1__bin_3__psf_2x2__interp_0.200"

        phase_tag = al.tagging.phase_tag_from_phase_settings(
            sub_size=1,
            transformer_class=al.TransformerDFT,
            real_space_shape_2d=(3, 3),
            real_space_pixel_scales=(1.0, 2.0),
            primary_beam_shape_2d=(2, 2),
        )

        assert (
            phase_tag == "phase_tag__dft__rs_shape_3x3__rs_pix_1.00x2.00__sub_1__pb_2x2"
        )


class TestPhaseTaggers:
    def test__positions_threshold_tagger(self):

        tag = al.tagging.positions_threshold_tag_from_positions_threshold(
            positions_threshold=None
        )
        assert tag == ""
        tag = al.tagging.positions_threshold_tag_from_positions_threshold(
            positions_threshold=1.0
        )
        assert tag == "__pos_1.00"
        tag = al.tagging.positions_threshold_tag_from_positions_threshold(
            positions_threshold=2.56
        )
        assert tag == "__pos_2.56"

    def test__sub_size_tagger(self):

        tag = al.tagging.sub_size_tag_from_sub_size(sub_size=1)
        assert tag == "__sub_1"
        tag = al.tagging.sub_size_tag_from_sub_size(sub_size=2)
        assert tag == "__sub_2"
        tag = al.tagging.sub_size_tag_from_sub_size(sub_size=4)
        assert tag == "__sub_4"

    def test__signal_to_noise_limit_tagger(self):

        tag = al.tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=None
        )
        assert tag == ""
        tag = al.tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=1
        )
        assert tag == "__snr_1"
        tag = al.tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=2
        )
        assert tag == "__snr_2"
        tag = al.tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=3
        )
        assert tag == "__snr_3"

    def test__bin_up_factor_tagger(self):

        tag = al.tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=None)
        assert tag == ""
        tag = al.tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=1)
        assert tag == ""
        tag = al.tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=2)
        assert tag == "__bin_2"
        tag = al.tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=3)
        assert tag == "__bin_3"

    def test__psf_shape_2d_tagger(self):

        tag = al.tagging.psf_shape_tag_from_psf_shape_2d(psf_shape_2d=None)
        assert tag == ""
        tag = al.tagging.psf_shape_tag_from_psf_shape_2d(psf_shape_2d=(2, 2))
        assert tag == "__psf_2x2"
        tag = al.tagging.psf_shape_tag_from_psf_shape_2d(psf_shape_2d=(3, 4))
        assert tag == "__psf_3x4"

    def test__pixel_scale_interpolation_grid_tagger(self):

        tag = al.tagging.pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
            pixel_scale_interpolation_grid=None
        )
        assert tag == ""
        tag = al.tagging.pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
            pixel_scale_interpolation_grid=0.5
        )
        assert tag == "__interp_0.500"
        tag = al.tagging.pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
            pixel_scale_interpolation_grid=0.25
        )
        assert tag == "__interp_0.250"
        tag = al.tagging.pixel_scale_interpolation_grid_tag_from_pixel_scale_interpolation_grid(
            pixel_scale_interpolation_grid=0.234
        )
        assert tag == "__interp_0.234"

    def test__transformer_tagger(self):
        tag = al.tagging.transformer_tag_from_transformer_class(
            transformer_class=al.TransformerDFT
        )
        assert tag == "__dft"
        tag = al.tagging.transformer_tag_from_transformer_class(
            transformer_class=al.TransformerFFT
        )
        assert tag == "__fft"
        tag = al.tagging.transformer_tag_from_transformer_class(
            transformer_class=al.TransformerNUFFT
        )
        assert tag == "__nufft"
        tag = al.tagging.transformer_tag_from_transformer_class(transformer_class=None)
        assert tag == ""

    def test__primary_beam_shape_2d_tagger(self):
        tag = al.tagging.primary_beam_shape_tag_from_primary_beam_shape_2d(
            primary_beam_shape_2d=None
        )
        assert tag == ""
        tag = al.tagging.primary_beam_shape_tag_from_primary_beam_shape_2d(
            primary_beam_shape_2d=(2, 2)
        )
        assert tag == "__pb_2x2"
        tag = al.tagging.primary_beam_shape_tag_from_primary_beam_shape_2d(
            primary_beam_shape_2d=(3, 4)
        )
        assert tag == "__pb_3x4"

    def test__real_space_shape_2d_tagger(self):

        tag = al.tagging.real_space_shape_2d_tag_from_real_space_shape_2d(
            real_space_shape_2d=None
        )
        assert tag == ""
        tag = al.tagging.real_space_shape_2d_tag_from_real_space_shape_2d(
            real_space_shape_2d=(2, 2)
        )
        assert tag == "__rs_shape_2x2"
        tag = al.tagging.real_space_shape_2d_tag_from_real_space_shape_2d(
            real_space_shape_2d=(3, 4)
        )
        assert tag == "__rs_shape_3x4"

    def test__real_space_pixel_scales_tagger(self):

        tag = al.tagging.real_space_pixel_scales_tag_from_real_space_pixel_scales(
            real_space_pixel_scales=None
        )
        assert tag == ""
        tag = al.tagging.real_space_pixel_scales_tag_from_real_space_pixel_scales(
            real_space_pixel_scales=(0.01, 0.02)
        )
        assert tag == "__rs_pix_0.01x0.02"
        tag = al.tagging.real_space_pixel_scales_tag_from_real_space_pixel_scales(
            real_space_pixel_scales=(2.0, 1.0)
        )
        assert tag == "__rs_pix_2.00x1.00"
