import autolens as al


def test__tag__mixture_of_values():

    settings = al.SettingsPhaseImaging(
        settings_masked_imaging=al.SettingsMaskedImaging(
            grid_class=al.Grid,
            grid_inversion_class=al.Grid,
            sub_size=2,
            signal_to_noise_limit=2,
            bin_up_factor=None,
            psf_shape_2d=None,
        ),
        settings_lens=al.SettingsLens(positions_threshold=2.0),
    )

    assert (
        settings.phase_tag_no_inversion == "settings__"
        "imaging[grid_sub_2__snr_2]__"
        "lens[pos_on]"
    )
    assert (
        settings.phase_tag_with_inversion == "settings__"
        "imaging[grid_sub_2_inv_sub_2__snr_2]__"
        "lens[pos_on]__"
        "pix[use_border]__"
        "inv[mat]"
    )

    settings = al.SettingsPhaseImaging(
        settings_masked_imaging=al.SettingsMaskedImaging(
            grid_class=al.GridIterate,
            grid_inversion_class=al.GridInterpolate,
            fractional_accuracy=0.5,
            pixel_scales_interp=0.3,
            bin_up_factor=3,
            psf_shape_2d=(2, 2),
        ),
        settings_lens=al.SettingsLens(
            positions_threshold=1.0, auto_positions_factor=0.5
        ),
        settings_pixelization=al.SettingsPixelization(
            use_border=False, is_stochastic=True
        ),
        log_likelihood_cap=200.01,
    )

    assert (
        settings.phase_tag_no_inversion == "settings__"
        "imaging[grid_facc_0.5__bin_3__psf_2x2]__"
        "lens[pos_on]__"
        "lh_cap_200.0"
    )
    assert (
        settings.phase_tag_with_inversion == "settings__"
        "imaging[grid_facc_0.5_inv_interp_0.300__bin_3__psf_2x2]__"
        "lens[pos_on]__"
        "pix[no_border__stochastic]__"
        "inv[mat]__"
        "lh_cap_200.0"
    )

    settings = al.SettingsPhaseInterferometer(
        settings_masked_interferometer=al.SettingsMaskedInterferometer(
            grid_class=al.GridIterate,
            grid_inversion_class=al.GridInterpolate,
            fractional_accuracy=0.5,
            pixel_scales_interp=0.3,
            transformer_class=al.TransformerDFT,
        ),
        settings_pixelization=al.SettingsPixelization(
            use_border=False, is_stochastic=True
        ),
        log_likelihood_cap=100.01,
    )

    assert (
        settings.phase_tag_no_inversion == "settings__"
        "interferometer[grid_facc_0.5__dft]__"
        "lens[pos_off]__"
        "lh_cap_100.0"
    )
    assert (
        settings.phase_tag_with_inversion == "settings__"
        "interferometer[grid_facc_0.5_inv_interp_0.300__dft]__"
        "lens[pos_off]__"
        "pix[no_border__stochastic]__"
        "inv[mat]__"
        "lh_cap_100.0"
    )

    settings = al.SettingsPhaseInterferometer(
        settings_masked_interferometer=al.SettingsMaskedInterferometer(
            transformer_class=al.TransformerNUFFT
        ),
        settings_inversion=al.SettingsInversion(use_linear_operators=True),
    )

    assert (
        settings.phase_tag_no_inversion == "settings__"
        "interferometer[grid_sub_2__nufft]__"
        "lens[pos_off]"
    )
    assert (
        settings.phase_tag_with_inversion == "settings__"
        "interferometer[grid_sub_2_inv_sub_2__nufft]__"
        "lens[pos_off]__"
        "pix[use_border]__"
        "inv[lop]"
    )
