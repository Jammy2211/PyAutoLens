import autolens as al


class TestPhaseTag:
    def test__mixture_of_values(self):

        settings = al.PhaseSettingsImaging(
            sub_size=2,
            signal_to_noise_limit=2,
            bin_up_factor=None,
            psf_shape_2d=None,
            positions_threshold=2.0,
            interpolation_pixel_scale=None,
        )

        assert settings.phase_tag == "phase_tag__sub_2__snr_2__pos_2.00"

        settings = al.PhaseSettingsImaging(
            sub_size=1,
            signal_to_noise_limit=None,
            bin_up_factor=3,
            psf_shape_2d=(2, 2),
            auto_positions_factor=0.5,
            positions_threshold=None,
            interpolation_pixel_scale=0.2,
        )

        assert (
            settings.phase_tag
            == "phase_tag__sub_1__interp_0.200__bin_3__psf_2x2__auto_pos_x0.50"
        )

        settings = al.PhaseSettingsInterferometer(
            sub_size=1,
            transformer_class=al.TransformerDFT,
            primary_beam_shape_2d=(2, 2),
        )

        assert settings.phase_tag == "phase_tag__dft__sub_1__pb_2x2"


class TestPhaseTaggers:
    def test__auto_positions_factor_tagger(self):

        settings = al.PhaseSettingsImaging(auto_positions_factor=None)
        assert settings.auto_positions_factor_tag == ""
        settings = al.PhaseSettingsImaging(auto_positions_factor=1.0)
        assert settings.auto_positions_factor_tag == "__auto_pos_x1.00"
        settings = al.PhaseSettingsImaging(auto_positions_factor=2.56)
        assert settings.auto_positions_factor_tag == "__auto_pos_x2.56"

    def test__positions_threshold_tagger(self):

        settings = al.PhaseSettingsImaging(positions_threshold=None)
        assert settings.positions_threshold_tag == ""
        settings = al.PhaseSettingsImaging(positions_threshold=1.0)
        assert settings.positions_threshold_tag == "__pos_1.00"
        settings = al.PhaseSettingsImaging(positions_threshold=2.56)
        assert settings.positions_threshold_tag == "__pos_2.56"

    def test__interpolation_pixel_scale_tagger(self):

        settings = al.PhaseSettingsImaging(interpolation_pixel_scale=None)
        assert settings.interpolation_pixel_scale_tag == ""
        settings = al.PhaseSettingsImaging(interpolation_pixel_scale=0.5)
        assert settings.interpolation_pixel_scale_tag == "__interp_0.500"
        settings = al.PhaseSettingsImaging(interpolation_pixel_scale=0.25)
        assert settings.interpolation_pixel_scale_tag == "__interp_0.250"
        settings = al.PhaseSettingsImaging(interpolation_pixel_scale=0.234)
        assert settings.interpolation_pixel_scale_tag == "__interp_0.234"


class TestEdit:
    def test__imaging__edit__changes_settings_if_input(self):

        settings = al.PhaseSettingsImaging(
            grid_class=al.Grid,
            grid_inversion_class=al.Grid,
            sub_size=2,
            fractional_accuracy=0.5,
            sub_steps=[2],
            signal_to_noise_limit=2,
            bin_up_factor=3,
            inversion_pixel_limit=100,
            psf_shape_2d=(3, 3),
            interpolation_pixel_scale=0.1,
            auto_positions_factor=2,
            positions_threshold=0.2,
            inversion_uses_border=False,
        )

        assert settings.grid_class is al.Grid
        assert settings.grid_inversion_class is al.Grid
        assert settings.sub_size == 2
        assert settings.fractional_accuracy == 0.5
        assert settings.sub_steps == [2]
        assert settings.signal_to_noise_limit == 2
        assert settings.bin_up_factor == 3
        assert settings.inversion_pixel_limit == 100
        assert settings.psf_shape_2d == (3, 3)
        assert settings.interpolation_pixel_scale == 0.1
        assert settings.auto_positions_factor == 2
        assert settings.positions_threshold == 0.2
        assert settings.inversion_uses_border == False

        settings = settings.edit(
            grid_class=al.GridIterator,
            grid_inversion_class=al.GridInterpolate,
            sub_steps=[5],
            inversion_pixel_limit=200,
            interpolation_pixel_scale=0.2,
            auto_positions_factor=3,
        )

        assert settings.grid_class is al.GridIterator
        assert settings.grid_inversion_class is al.GridInterpolate
        assert settings.sub_size == 2
        assert settings.fractional_accuracy == 0.5
        assert settings.sub_steps == [5]
        assert settings.signal_to_noise_limit == 2
        assert settings.bin_up_factor == 3
        assert settings.inversion_pixel_limit == 200
        assert settings.psf_shape_2d == (3, 3)
        assert settings.interpolation_pixel_scale == 0.2
        assert settings.auto_positions_factor == 3
        assert settings.positions_threshold == 0.2
        assert settings.inversion_uses_border == False

        settings = settings.edit(
            sub_size=3,
            fractional_accuracy=0.7,
            signal_to_noise_limit=4,
            bin_up_factor=5,
            psf_shape_2d=(5, 5),
            positions_threshold=0.4,
            inversion_uses_border=True,
        )

        assert settings.grid_class is al.GridIterator
        assert settings.grid_inversion_class is al.GridInterpolate
        assert settings.sub_size == 3
        assert settings.fractional_accuracy == 0.7
        assert settings.sub_steps == [5]
        assert settings.signal_to_noise_limit == 4
        assert settings.bin_up_factor == 5
        assert settings.inversion_pixel_limit == 200
        assert settings.psf_shape_2d == (5, 5)
        assert settings.interpolation_pixel_scale == 0.2
        assert settings.auto_positions_factor == 3
        assert settings.positions_threshold == 0.4
        assert settings.inversion_uses_border == True

    def test__interferometer__edit__changes_settings_if_input(self):

        settings = al.PhaseSettingsInterferometer(
            grid_class=al.Grid,
            grid_inversion_class=al.Grid,
            sub_size=2,
            fractional_accuracy=0.5,
            sub_steps=[2],
            signal_to_noise_limit=2,
            bin_up_factor=3,
            inversion_pixel_limit=100,
            transformer_class=al.TransformerDFT,
            primary_beam_shape_2d=(3, 3),
            interpolation_pixel_scale=0.1,
            auto_positions_factor=2,
            positions_threshold=0.2,
            inversion_uses_border=False,
        )

        assert settings.grid_class is al.Grid
        assert settings.grid_inversion_class is al.Grid
        assert settings.sub_size == 2
        assert settings.fractional_accuracy == 0.5
        assert settings.sub_steps == [2]
        assert settings.signal_to_noise_limit == 2
        assert settings.bin_up_factor == 3
        assert settings.inversion_pixel_limit == 100
        assert settings.transformer_class is al.TransformerDFT
        assert settings.primary_beam_shape_2d == (3, 3)
        assert settings.interpolation_pixel_scale == 0.1
        assert settings.auto_positions_factor == 2
        assert settings.positions_threshold == 0.2
        assert settings.inversion_uses_border == False

        settings = settings.edit(
            grid_class=al.GridIterator,
            grid_inversion_class=al.GridInterpolate,
            sub_steps=[5],
            inversion_pixel_limit=200,
            transformer_class=al.TransformerFFT,
            interpolation_pixel_scale=0.2,
            auto_positions_factor=3,
        )

        assert settings.grid_class is al.GridIterator
        assert settings.grid_inversion_class is al.GridInterpolate
        assert settings.sub_size == 2
        assert settings.fractional_accuracy == 0.5
        assert settings.sub_steps == [5]
        assert settings.signal_to_noise_limit == 2
        assert settings.bin_up_factor == 3
        assert settings.inversion_pixel_limit == 200
        assert settings.transformer_class is al.TransformerFFT
        assert settings.primary_beam_shape_2d == (3, 3)
        assert settings.interpolation_pixel_scale == 0.2
        assert settings.auto_positions_factor == 3
        assert settings.positions_threshold == 0.2
        assert settings.inversion_uses_border == False

        settings = settings.edit(
            sub_size=3,
            fractional_accuracy=0.7,
            signal_to_noise_limit=4,
            bin_up_factor=5,
            primary_beam_shape_2d=(5, 5),
            positions_threshold=0.4,
            inversion_uses_border=True,
        )

        assert settings.grid_class is al.GridIterator
        assert settings.grid_inversion_class is al.GridInterpolate
        assert settings.sub_size == 3
        assert settings.fractional_accuracy == 0.7
        assert settings.sub_steps == [5]
        assert settings.signal_to_noise_limit == 4
        assert settings.bin_up_factor == 5
        assert settings.inversion_pixel_limit == 200
        assert settings.transformer_class is al.TransformerFFT
        assert settings.primary_beam_shape_2d == (5, 5)
        assert settings.interpolation_pixel_scale == 0.2
        assert settings.auto_positions_factor == 3
        assert settings.positions_threshold == 0.4
        assert settings.inversion_uses_border == True
