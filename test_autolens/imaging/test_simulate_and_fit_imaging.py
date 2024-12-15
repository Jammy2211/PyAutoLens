import os
from os import path
import shutil

import autolens as al
import numpy as np
import pytest


def test__perfect_fit__chi_squared_0():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2)

    psf = al.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.lp.Exponential(centre=(0.1, 0.1), intensity=0.5)
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "data_temp",
        "simulate_and_fit",
    )

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if path.exists(file_path) is False:
        os.makedirs(file_path)

    dataset.output_to_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
    )

    dataset = al.Imaging.from_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        pixel_scales=0.2,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.8
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    assert fit.chi_squared == pytest.approx(0.0, 1e-4)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) is True:
        shutil.rmtree(file_path)



def test__simulate_imaging_data_and_fit__known_likelihood():

    grid = al.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2)

    psf = al.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(16, 16)),
        regularization=al.reg.Constant(coefficient=(1.0)),
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        disk=al.lp.Sersic(centre=(0.2, 0.2), intensity=0.2),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy_0 = al.Galaxy(redshift=1.0, pixelization=pixelization)
    source_galaxy_1 = al.Galaxy(redshift=2.0, pixelization=pixelization)
    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
    )

    simulator = al.SimulatorImaging(exposure_time=300.0, psf=psf, noise_seed=1)

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=2.0
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    assert fit.figure_of_merit == pytest.approx(526.353910, 1.0e-2)


def test__simulate_imaging_data_and_fit__linear_light_profiles_agree_with_standard_light_profiles():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2)

    psf = al.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.sub_size = 1
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.8
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy_linear = al.Galaxy(
        redshift=1.0,
        bulge=al.lp_linear.Sersic(sersic_index=1.0),
        disk=al.lp_linear.Sersic(sersic_index=4.0),
    )

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_linear]
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array([0.1, 0.1, 0.2]), 1.0e-4
    )
    assert fit_linear.linear_light_profile_intensity_dict[
        lens_galaxy_linear.light
    ] == pytest.approx(0.1, 1.0e-2)
    assert fit_linear.linear_light_profile_intensity_dict[
        source_galaxy_linear.bulge
    ] == pytest.approx(0.1, 1.0e-2)
    assert fit_linear.linear_light_profile_intensity_dict[
        source_galaxy_linear.disk
    ] == pytest.approx(0.2, 1.0e-2)
    assert fit.log_likelihood == fit_linear.figure_of_merit
    assert fit_linear.figure_of_merit == pytest.approx(-45.02798, 1.0e-4)

    lens_galaxy_image = lens_galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.uniform,
        convolver=masked_dataset.convolver,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image, 1.0e-4
    )
    assert fit_linear.model_images_of_planes_list[0] == pytest.approx(
        lens_galaxy_image, 1.0e-4
    )

    traced_grid_2d_list = tracer.traced_grid_2d_list_from(grid=masked_dataset.grids.uniform)
    traced_blurring_grid_2d_list = tracer.traced_grid_2d_list_from(
        grid=masked_dataset.grids.blurring
    )

    source_galaxy_image = source_galaxy.blurred_image_2d_from(
        grid=traced_grid_2d_list[1],
        convolver=masked_dataset.convolver,
        blurring_grid=traced_blurring_grid_2d_list[1],
    )

    assert fit_linear.galaxy_model_image_dict[source_galaxy_linear] == pytest.approx(
        source_galaxy_image, 1.0e-4
    )

    assert fit_linear.model_images_of_planes_list[1] == pytest.approx(
        source_galaxy_image, 1.0e-4
    )


def test__simulate_imaging_data_and_fit__linear_light_profiles_and_pixelization():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2)

    psf = al.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=100.0),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.sub_size = 1
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.8
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    # pixelization = al.Pixelization(
    #     image_mesh=al.image_mesh.Overlay(shape=(3,3)),
    #     mesh=al.mesh.Delaunay(),
    #     regularization=al.reg.Constant(coefficient=0.01),
    # )

    source_galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_pix]
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array(
            [
                99.98206689,
                0.12232328,
                0.10729026,
                0.10243489,
                0.15267803,
                0.13052345,
                0.10758802,
                0.1491073,
                0.15455053,
                0.12146136,
            ]
        ),
        1.0e-4,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-84.04875317, 1.0e-4)

    lens_galaxy_image = lens_galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.uniform,
        convolver=masked_dataset.convolver,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image, 1.0e-2
    )
    assert fit_linear.model_images_of_planes_list[0] == pytest.approx(
        lens_galaxy_image, 1.0e-2
    )

    assert fit_linear.galaxy_model_image_dict[source_galaxy_pix][0] == pytest.approx(
        0.063911, 1.0e-4
    )

    assert fit_linear.model_images_of_planes_list[1][0] == pytest.approx(
        0.063911, 1.0e-4
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings_inversion=al.SettingsInversion(
            use_w_tilde=False,
            use_positive_only_solver=True,
            force_edge_pixels_to_zeros=True
        ),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array(
            [
                100.01548,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        1.0e-4,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-84.11166, 1.0e-4)

    pixelization = al.Pixelization(
        image_mesh=al.image_mesh.Overlay(shape=(3,3)),
        mesh=al.mesh.Delaunay(),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    source_galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_pix]
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    assert fit_linear.figure_of_merit == pytest.approx(-73.27676850869975, 1.0e-4)


def test__simulate_imaging_data_and_fit__linear_light_profiles_and_pixelization__sub_2():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, over_sampling_size=2)

    psf = al.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.Sersic(centre=(0.1, 0.1), intensity=100.0),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=al.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.8
    )

    dataset = al.Imaging(
        data=dataset.data,
        psf=dataset.psf,
        noise_map=dataset.noise_map,
        over_sampling=al.OverSamplingDataset(
            uniform=al.OverSampling(sub_size=2),
            pixelization=al.OverSampling(sub_size=2)
        )
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    source_galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_pix]
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )


    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array(
            [
                99.95778211,  0.26760714,  0.18616947,  0.1814436,   0.35814812,  0.2721518,
                0.18617081,  0.39590081,  0.35814604,  0.26760716
            ]
        ),
        1.0e-4,
    )

    assert fit_linear.figure_of_merit == pytest.approx(-84.36224277776512, 1.0e-4)

    lens_galaxy_image = lens_galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.uniform,
        convolver=masked_dataset.convolver,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image, 1.0e-2
    )
    assert fit_linear.model_images_of_planes_list[0] == pytest.approx(
        lens_galaxy_image, 1.0e-2
    )

    assert fit_linear.galaxy_model_image_dict[source_galaxy_pix][0] == pytest.approx(
        0.1473073435555056, 1.0e-4
    )

    assert fit_linear.model_images_of_planes_list[1][0] == pytest.approx(
        0.147307343555, 1.0e-4
    )

    assert fit_linear.subtracted_images_of_planes_list[1][0] == pytest.approx(
        0.35652124420455, 1.0e-4
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings_inversion=al.SettingsInversion(
            use_w_tilde=False,
            use_positive_only_solver=True,
            force_edge_pixels_to_zeros=True
        ),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array(
            [
                100.01548,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        1.0e-4,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-84.66302233089499, 1.0e-4)


def test__simulate_imaging_data_and_fit__complex_fit_compare_mapping_matrix_w_tilde():

    grid = al.Grid2D.uniform(shape_native=(21, 21), pixel_scales=0.1)

    psf = al.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_0 = al.Galaxy(
        redshift=0.1,
        light=al.lp.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    lens_1 = al.Galaxy(
        redshift=0.2,
        light=al.lp.Sersic(centre=(0.2, 0.2)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    lens_2 = al.Galaxy(
        redshift=0.3,
        light=al.lp.Sersic(centre=(0.3, 0.3)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    source_0 = al.Galaxy(redshift=0.4, bulge=al.lp.Sersic(centre=(0.3, 0.3)))
    source_1 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(centre=(0.3, 0.3)))
    tracer = al.Tracer(galaxies=[lens_0, lens_1, lens_2, source_0, source_1])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.sub_size = 2
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.8
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    lens_0 = al.Galaxy(
        redshift=0.1,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    lens_1 = al.Galaxy(
        redshift=0.2,
        light=al.lp_linear.Sersic(centre=(0.2, 0.2)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    lens_2 = al.Galaxy(
        redshift=0.3,
        light=al.lp_linear.Sersic(centre=(0.3, 0.3)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source_0 = al.Galaxy(redshift=0.4, pixelization=pixelization)

    pixelization = al.Pixelization(
        mesh=al.mesh.Rectangular(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source_1 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(
        galaxies=[lens_0, lens_1, lens_2, source_0, source_1]
    )

    fit_mapping = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
    )

    fit_w_tilde = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=True),
    )

    assert fit_mapping.inversion.curvature_matrix == pytest.approx(
            fit_w_tilde.inversion.curvature_matrix,
        1.0e-4,
    )

    assert fit_mapping.inversion.regularization_matrix == pytest.approx(
            fit_w_tilde.inversion.regularization_matrix,
        1.0e-4,
    )

    preloads = al.Preloads(
        linear_func_operated_mapping_matrix_dict=fit_mapping.inversion.linear_func_operated_mapping_matrix_dict,
        data_linear_func_matrix_dict=fit_mapping.inversion.data_linear_func_matrix_dict
    )

    fit_w_tilde = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer,
        preloads=preloads,
        settings_inversion=al.SettingsInversion(use_w_tilde=True),
    )

    assert fit_mapping.inversion.curvature_matrix == pytest.approx(
            fit_w_tilde.inversion.curvature_matrix,
        1.0e-4,
    )

    assert fit_mapping.inversion.regularization_matrix == pytest.approx(
            fit_w_tilde.inversion.regularization_matrix,
        1.0e-4,
    )

    preloads = al.Preloads(
        mapper_operated_mapping_matrix_dict=fit_mapping.inversion.mapper_operated_mapping_matrix_dict,
    )

    fit_w_tilde = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer,
        preloads=preloads,
        settings_inversion=al.SettingsInversion(use_w_tilde=True),
    )

    assert fit_mapping.inversion.curvature_matrix == pytest.approx(
            fit_w_tilde.inversion.curvature_matrix,
        1.0e-4,
    )

    assert fit_mapping.inversion.regularization_matrix == pytest.approx(
            fit_w_tilde.inversion.regularization_matrix,
        1.0e-4,
    )


def test__perfect_fit__chi_squared_0__non_uniform_over_sampling():

    over_sampling = al.OverSampling(sub_size=8)

    grid = al.Grid2D.uniform(
        shape_native=(31, 31),
        pixel_scales=0.2,
        over_sampling=over_sampling
    )

    psf = al.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.3),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.lp.Sersic(centre=(0.1, 0.1), intensity=0.5, sersic_index=1.2)
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "data_temp",
        "simulate_and_fit",
    )

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if path.exists(file_path) is False:
        os.makedirs(file_path)

    dataset.output_to_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
    )

    dataset = al.Imaging.from_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        pixel_scales=0.2,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=1.5
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    traced_grid = tracer.traced_grid_2d_list_from(
        grid=masked_dataset.grids.uniform,
    )[-1]

    masked_dataset = masked_dataset.apply_over_sampling(
        over_sampling=al.OverSamplingDataset(
            uniform=al.OverSampling(sub_size=1),
            non_uniform=al.OverSampling.over_sample_size_via_radial_bins_from(
            grid=traced_grid, sub_size_list=[8, 2], radial_list=[0.3], centre_list=[source_galaxy.light.centre]
        ))
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    assert fit.chi_squared < 0.1

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) is True:
        shutil.rmtree(file_path)


def test__fit_figure_of_merit__mge_mass_model(masked_imaging_7x7, masked_imaging_covariance_7x7):
    over_sampling = al.OverSampling(sub_size=8)

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2,
                             over_sampling=over_sampling)

    psf = al.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light_0=al.lp.Gaussian(intensity=1.0),
        light_1=al.lp.Gaussian(intensity=2.0),
        mass_0=al.mp.Gaussian(intensity=1.0, mass_to_light_ratio=3.0),
        mass_1=al.mp.Gaussian(intensity=2.0, mass_to_light_ratio=4.0),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0, light=al.lp.Exponential(centre=(0.1, 0.1), intensity=0.5)
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    dataset = al.SimulatorImaging(exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False)

    dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "data_temp",
        "simulate_and_fit",
    )

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if path.exists(file_path) is False:
        os.makedirs(file_path)

    dataset.output_to_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
    )

    dataset = al.Imaging.from_fits(
        data_path=path.join(file_path, "data.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        pixel_scales=0.2,
        over_sampling=al.OverSamplingDataset(uniform=over_sampling)
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.8
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lmp.Gaussian(intensity=1.0, mass_to_light_ratio=3.0),
            al.lmp.Gaussian(intensity=2.0, mass_to_light_ratio=4.0),
        ]
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        bulge=basis,
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    # The value is actually not zero before the blurring grid assumes a sub_size=1
    # and does not use the iterative grid, which has a small impact on the chi-squared

    assert fit.chi_squared == pytest.approx(5.706423629698664e-05, 1e-4)

    over_sampling = al.OverSampling(sub_size=8)

    masked_dataset = masked_dataset.apply_over_sampling(
        al.OverSamplingDataset(uniform=over_sampling)
    )

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lmp_linear.Gaussian(intensity=1.0, mass_to_light_ratio=3.0),
            al.lmp_linear.Gaussian(intensity=2.0, mass_to_light_ratio=4.0),
        ]
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        bulge=basis,
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    # The value is actually not zero before the blurring grid assumes a sub_size=1
    # and does not use the iterative grid, which has a small impact on the chi-squared

    assert fit.chi_squared == pytest.approx(3.295535243634485e-05, 1e-4)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) is True:
        shutil.rmtree(file_path)