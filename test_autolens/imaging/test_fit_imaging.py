import copy
import numpy as np
import pytest

import autolens as al


def test__model_image__with_and_without_psf_blurring(
    masked_imaging_7x7_no_blur, masked_imaging_7x7
):

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.m.MockLightProfile(image_2d_value=1.0, image_2d_first_value=2.0),
    )
    tracer = al.Tracer(galaxies=[g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.model_data.slim == pytest.approx(
        np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(-14.6337, 1.0e-4)

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.model_data.slim == pytest.approx(
        np.array([1.33, 1.16, 1.0, 1.16, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-14.52960, 1.0e-4)



def test__fit_figure_of_merit(masked_imaging_7x7, masked_imaging_covariance_7x7):

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-648.4814555620, 1.0e-4)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
            al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        ]
    )

    g0 = al.Galaxy(
        redshift=0.5, bulge=basis, mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0)
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-648.4814555620, 1.0e-4)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    g0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.9122020414322, 1.0e-4)

    galaxy_light = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[galaxy_light, galaxy_pix])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-29.2344033729098, 1.0e-4)

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        disk=al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )

    tracer = al.Tracer(galaxies=[g0_linear, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-16.9731347648, 1.0e-4)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
            al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
        ]
    )

    g0_linear = al.Galaxy(
        redshift=0.5, bulge=basis, mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0)
    )

    tracer = al.Tracer(galaxies=[g0_linear, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-16.97313476, 1.0e-4)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
            al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
        ],
        regularization=al.reg.Constant(coefficient=1.0),
    )

    g0_basis = al.Galaxy(
        redshift=0.5, bulge=basis, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
    )

    tracer = al.Tracer(galaxies=[g0_basis, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-51.0835080747, 1.0e-4)

    tracer = al.Tracer(galaxies=[g0_linear, galaxy_pix])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.79906, 1.0e-4)

    g0_operated = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )

    g1_operated = al.Galaxy(redshift=1.0, bulge=al.lp_operated.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0_operated, g1_operated])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-745.25961066, 1.0e-4)

    g0_linear_operated = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear_operated.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )

    g1_linear_operated = al.Galaxy(
        redshift=1.0, bulge=al.lp_linear_operated.Sersic(centre=(0.05, 0.05), sersic_index=4.0)
    )

    tracer = al.Tracer(galaxies=[g0_linear_operated, g1_linear_operated])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-14.933306470, 1.0e-4)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_covariance_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-775.003133428, 1.0e-4)


def test__fit_figure_of_merit__sub_2(image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7, masked_imaging_covariance_7x7):

    dataset = al.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
    )

    masked_imaging_7x7 = dataset.apply_mask(
        mask=mask_2d_7x7
    )

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        disk=al.lp.Sersic(intensity=2.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-41.60614104506, 1.0e-4)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp.Sersic(intensity=1.0),
            al.lp.Sersic(intensity=2.0),
        ]
    )

    g0 = al.Galaxy(
        redshift=0.5, bulge=basis, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-41.60614104506277, 1.0e-4)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    g0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[al.Galaxy(redshift=0.5), g0])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-22.912202041432, 1.0e-4)

    galaxy_light = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[galaxy_light, galaxy_pix])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-23.040374194334518, 1.0e-4)

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(sersic_index=1.0),
        disk=al.lp_linear.Sersic(sersic_index=4.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    tracer = al.Tracer(galaxies=[g0_linear, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-18.47282483, 1.0e-4)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Sersic(sersic_index=1.0),
            al.lp_linear.Sersic(sersic_index=4.0),
        ]
    )

    g0_linear = al.Galaxy(
        redshift=0.5, bulge=basis, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
    )

    tracer = al.Tracer(galaxies=[g0_linear, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-18.4728248395, 1.0e-4)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp_linear.Sersic(sersic_index=1.0),
            al.lp_linear.Sersic(sersic_index=4.0),
        ],
        regularization=al.reg.Constant(coefficient=1.0),
    )

    g0_basis = al.Galaxy(
        redshift=0.5, bulge=basis, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
    )

    tracer = al.Tracer(galaxies=[g0_basis, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-27.8563999226980, 1.0e-4)

    tracer = al.Tracer(galaxies=[g0_linear, galaxy_pix])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx( -22.8021113243233, 1.0e-4)

    g0_operated = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1_operated = al.Galaxy(redshift=1.0, bulge=al.lp_operated.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0_operated, g1_operated])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-51.32884812, 1.0e-4)

    g0_linear_operated = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear_operated.Sersic(sersic_index=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1_linear_operated = al.Galaxy(
        redshift=1.0, bulge=al.lp_linear_operated.Sersic(sersic_index=4.0)
    )

    tracer = al.Tracer(galaxies=[g0_linear_operated, g1_linear_operated])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-14.532352248, 1.0e-4)

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_covariance_7x7, tracer=tracer)

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-775.0031334280, 1.0e-4)


def test__fit__sky___handles_special_behaviour(masked_imaging_7x7):
    masked_imaging_7x7 = copy.copy(masked_imaging_7x7)

    masked_imaging_7x7.data -= 100.0

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(0.05, 0.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    fit = al.FitImaging(
        dataset=masked_imaging_7x7, tracer=tracer, dataset_model=al.DatasetModel(background_sky_level=5.0)
    )

    assert fit.figure_of_merit == pytest.approx(-18050.8847818, 1.0e-4)


def test__fit__model_dataset__grid_offset__handles_special_behaviour(masked_imaging_7x7):

    # Numerical value changes slightly due to prevision in grid values

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(-1.05, -2.05), intensity=1.0),
        disk=al.lp.Sersic(centre=(-1.05, -2.05), intensity=2.0),
        mass_profile=al.mp.IsothermalSph(centre=(-1.05, -2.05), einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0,
                   bulge=al.lp.Sersic(centre=(-1.05, -2.05), intensity=1.0)
                   )

    tracer = al.Tracer(galaxies=[g0, g1])

    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        dataset_model=al.DatasetModel(grid_offset=(1.0, 2.0))
    )

    assert fit.figure_of_merit == pytest.approx(-648.48145556, 1.0e-4)

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(centre=(-1.0, -2.0), sersic_index=1.0),
        disk=al.lp_linear.Sersic(centre=(-1.0, -2.0), sersic_index=4.0),
        mass_profile=al.mp.IsothermalSph(centre=(-1.0, -2.0), einstein_radius=1.0),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0_linear, galaxy_pix])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer,
                        dataset_model=al.DatasetModel(grid_offset=(1.0, 2.0))
                        )
    assert fit.figure_of_merit == pytest.approx(-22.8021112977481, 1.0e-4)


def test__galaxy_model_image_dict(masked_imaging_7x7):

    # Normal Light Profiles Only

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))
    g2 = al.Galaxy(redshift=1.0)

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    blurred_image_2d_list = tracer.blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grids.lp,
        psf=masked_imaging_7x7.psf,
        blurring_grid=masked_imaging_7x7.grids.blurring,
    )

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(
        blurred_image_2d_list[0].array, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[g1] == pytest.approx(
        blurred_image_2d_list[1].array, 1.0e-4
    )
    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()

    assert fit.model_data.native == pytest.approx(
        fit.galaxy_model_image_dict[g0].native.array + fit.galaxy_model_image_dict[g1].native.array,
        1.0e-4,
    )

    # Linear Light Profiles Only

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1_linear = al.Galaxy(redshift=1.0, bulge=al.lp_linear.Sersic())

    tracer = al.Tracer(galaxies=[g0_linear, g1_linear, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(
        -4.99645959e-01, 1.0e-2
    )
    assert fit.galaxy_model_image_dict[g1_linear][4] == pytest.approx(
        1.9986499980, 1.0e-2
    )
    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()

    assert fit.model_data == pytest.approx(
        fit.galaxy_model_image_dict[g0_linear].array + fit.galaxy_model_image_dict[g1_linear].array,
        1.0e-4,
    )

    # Pixelization + Regularizaiton only

    g0_no_light = al.Galaxy(
        redshift=0.5, mass_profile=al.mp.IsothermalSph(einstein_radius=1.0)
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0_no_light, galaxy_pix_0])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert (fit.galaxy_model_image_dict[g0_no_light].array == np.zeros(9)).all()
    assert fit.galaxy_model_image_dict[galaxy_pix_0].array[4] == pytest.approx(
        1.259965886, 1.0e-4
    )

    assert fit.model_data == pytest.approx(
        fit.galaxy_model_image_dict[galaxy_pix_0].array, 1.0e-4
    )

    # Normal light + Linear Light PRofiles + Pixelization + Regularizaiton

    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(
        galaxies=[g0, g0_linear, g2, galaxy_pix_0, galaxy_pix_1]
    )

    masked_imaging_7x7.data[0] = 3.0

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer, settings_inversion=al.SettingsInversion(use_w_tilde=False))

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(
        blurred_image_2d_list[0].array, 1.0e-4
    )

    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(
        -9.31143037, 1.0e-4
    )

    assert fit.galaxy_model_image_dict[galaxy_pix_0][4] == pytest.approx(
        0.94918443, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[galaxy_pix_1][4] == pytest.approx(
        0.94918442, 1.0e-4
    )
    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()


def test__subtracted_image_of_galaxies_dict(masked_imaging_7x7):

    # 2 Planes with Summed Galaxies

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=2.0))
    g2 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    g0_image = g0.blurred_image_2d_from(
        grid=masked_imaging_7x7.grids.lp,
        blurring_grid=masked_imaging_7x7.grids.blurring,
        psf=masked_imaging_7x7.psf
    )

    g1_image = g1.blurred_image_2d_from(
        grid=masked_imaging_7x7.grids.lp,
        blurring_grid=masked_imaging_7x7.grids.blurring,
        psf=masked_imaging_7x7.psf
    )

    g2_image = g2.blurred_image_2d_from(
        grid=masked_imaging_7x7.grids.lp,
        blurring_grid=masked_imaging_7x7.grids.blurring,
        psf=masked_imaging_7x7.psf
    )

    assert fit.subtracted_images_of_galaxies_dict[g0] == pytest.approx(
        masked_imaging_7x7.data.array - g1_image.array - g2_image.array, 1.0e-4
    )
    assert fit.subtracted_images_of_galaxies_dict[g1] == pytest.approx(
        masked_imaging_7x7.data.array - g0_image.array - g2_image.array, 1.0e-4
    )
    assert fit.subtracted_images_of_galaxies_dict[g2] == pytest.approx(
        masked_imaging_7x7.data.array - g0_image.array - g1_image.array, 1.0e-4
    )

    # 3 Planes

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )
    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=2.0))
    g2 = al.Galaxy(redshift=2.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    blurred_image_2d_list = tracer.blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grids.lp,
        psf=masked_imaging_7x7.psf,
        blurring_grid=masked_imaging_7x7.grids.blurring,
    )

    assert fit.subtracted_images_of_galaxies_dict[g0] == pytest.approx(
        masked_imaging_7x7.data.array - blurred_image_2d_list[1].array - blurred_image_2d_list[2].array, 1.0e-4
    )
    assert fit.subtracted_images_of_galaxies_dict[g1] == pytest.approx(
        masked_imaging_7x7.data.array - blurred_image_2d_list[0].array - blurred_image_2d_list[2].array, 1.0e-4
    )
    assert fit.subtracted_images_of_galaxies_dict[g2] == pytest.approx(
        masked_imaging_7x7.data.array - blurred_image_2d_list[0].array - blurred_image_2d_list[1].array, 1.0e-4
    )


def test__model_images_of_planes_list(masked_imaging_7x7_sub_2):

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1_linear = al.Galaxy(redshift=0.75, bulge=al.lp_linear.Sersic())

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    galaxy_pix_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)
    galaxy_pix_1 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(
        galaxies=[g0, g1_linear, galaxy_pix_0, galaxy_pix_1]
    )

    fit = al.FitImaging(dataset=masked_imaging_7x7_sub_2, tracer=tracer, settings_inversion=al.SettingsInversion(use_w_tilde=False))

    assert fit.model_images_of_planes_list[0] == pytest.approx(
        fit.galaxy_model_image_dict[g0].array, 1.0e-4
    )
    assert fit.model_images_of_planes_list[1] == pytest.approx(
        fit.galaxy_model_image_dict[g1_linear].array, 1.0e-4
    )
    assert fit.model_images_of_planes_list[2] == pytest.approx(
        fit.galaxy_model_image_dict[galaxy_pix_0].array
        + fit.galaxy_model_image_dict[galaxy_pix_1].array,
        1.0e-4,
    )

    assert fit.model_images_of_planes_list[2][0] == pytest.approx(1.56110288, 1.0e-4)


def test__subtracted_images_of_planes_list(masked_imaging_7x7_no_blur, masked_imaging_7x7_no_blur_sub_2):

    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))

    g1 = al.Galaxy(redshift=0.75, bulge=al.lp.Sersic(intensity=2.0))

    g2 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.subtracted_images_of_planes_list[0].slim[0] == pytest.approx(0.200638, 1.0e-4)
    assert fit.subtracted_images_of_planes_list[1].slim[0] == pytest.approx(0.360511, 1.0e-4)
    assert fit.subtracted_images_of_planes_list[2].slim[0] == pytest.approx(0.520383, 1.0e-4)

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur_sub_2, tracer=tracer)

    assert fit.subtracted_images_of_planes_list[2].slim[0] == pytest.approx(0.475542485138, 1.0e-4)

    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(intensity=1.0))

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=2.0))

    g2 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=3.0))

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7_no_blur, tracer=tracer)

    assert fit.subtracted_images_of_planes_list[0].slim[0] == pytest.approx(0.200638, 1.0e-4)
    assert fit.subtracted_images_of_planes_list[1].slim[0] == pytest.approx(0.840127, 1.0e-4)


def test___unmasked_blurred_images(masked_imaging_7x7):

    g0 = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(intensity=1.0),
        mass_profile=al.mp.IsothermalSph(einstein_radius=1.0),
    )

    g1 = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(intensity=1.0))

    tracer = al.Tracer(galaxies=[g0, g1])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    blurred_images_of_planes = tracer.blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grids.lp,
        psf=masked_imaging_7x7.psf,
        blurring_grid=masked_imaging_7x7.grids.blurring,
    )

    unmasked_blurred_image = tracer.unmasked_blurred_image_2d_from(
        grid=masked_imaging_7x7.grids.lp, psf=masked_imaging_7x7.psf
    )

    assert (fit.unmasked_blurred_image == unmasked_blurred_image).all()

    unmasked_blurred_image_of_planes_list = tracer.unmasked_blurred_image_2d_list_from(
        grid=masked_imaging_7x7.grids.lp, psf=masked_imaging_7x7.psf
    )

    assert (
        unmasked_blurred_image_of_planes_list[0]
        == fit.unmasked_blurred_image_of_planes_list[0]
    ).all()
    assert (
        unmasked_blurred_image_of_planes_list[1]
        == fit.unmasked_blurred_image_of_planes_list[1]
    ).all()


def test__tracer_linear_light_profiles_to_light_profiles(masked_imaging_7x7):

    g0 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(centre=(0.05, 0.05), intensity=1.0))

    g0_linear = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        mass_profile=al.mp.IsothermalSph(centre=(0.05, 0.05), einstein_radius=1.0),
    )

    g1_linear = al.Galaxy(redshift=1.0, bulge=al.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0))


    tracer = al.Tracer(galaxies=[g0, g0_linear, g1_linear])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.tracer.galaxies[0].bulge.intensity == pytest.approx(1.0, 1.0e-4)

    tracer = fit.tracer_linear_light_profiles_to_light_profiles

    assert tracer.galaxies[0].bulge.intensity == pytest.approx(1.0, 1.0e-4)
    assert tracer.galaxies[1].bulge.intensity == pytest.approx(-5.830442986, 1.0e-4)
    assert tracer.galaxies[2].bulge.intensity == pytest.approx(0.135755913, 1.0e-4)



def test__total_mappers(masked_imaging_7x7):

    g0 = al.Galaxy(redshift=0.5)

    g1 = al.Galaxy(redshift=1.0)

    g2 = al.Galaxy(redshift=2.0)

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.total_mappers == 0

    pixelization = al.Pixelization(mesh=al.mesh.RectangularUniform())

    g2 = al.Galaxy(redshift=2.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    fit = al.FitImaging(dataset=masked_imaging_7x7, tracer=tracer)

    assert fit.total_mappers == 1

    g0 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    g1 = al.Galaxy(redshift=1.0, pixelization=pixelization)

    g2 = al.Galaxy(redshift=2.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[g0, g1, g2])

    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit.total_mappers == 3
