import os
from os import path
import shutil

import autolens as al
import numpy as np
import pytest


def test__perfect_fit__chi_squared_0():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, over_sample_size=1)

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
        over_sample_size_lp=1
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
        mesh=al.mesh.RectangularUniform(shape=(16, 16)),
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
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=2.005
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    fit = al.FitImaging(dataset=masked_dataset, tracer=tracer)

    assert fit.figure_of_merit == pytest.approx(574.3397424970, 1.0e-2)


def test__simulate_imaging_data_and_fit__linear_light_profiles_agree_with_standard_light_profiles():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, over_sample_size=1)

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
    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.805
    )

    masked_dataset = dataset.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_over_sampling(
        over_sample_size_lp=1
    )

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
        grid=masked_dataset.grids.lp,
        psf=masked_dataset.psf,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image.array, 1.0e-4
    )
    assert fit_linear.model_images_of_planes_list[0] == pytest.approx(
        lens_galaxy_image.array, 1.0e-4
    )

    traced_grid_2d_list = tracer.traced_grid_2d_list_from(grid=masked_dataset.grids.lp)
    traced_blurring_grid_2d_list = tracer.traced_grid_2d_list_from(
        grid=masked_dataset.grids.blurring
    )

    source_galaxy_image = source_galaxy.blurred_image_2d_from(
        grid=traced_grid_2d_list[1],
        psf=masked_dataset.psf,
        blurring_grid=traced_blurring_grid_2d_list[1],
    )

    assert fit_linear.galaxy_model_image_dict[source_galaxy_linear] == pytest.approx(
        source_galaxy_image.array, 1.0e-4
    )

    assert fit_linear.model_images_of_planes_list[1] == pytest.approx(
        source_galaxy_image.array, 1.0e-4
    )


def test__simulate_imaging_data_and_fit__linear_light_profiles_and_pixelization():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, over_sample_size=1)

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
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.805
    )

    masked_dataset = dataset.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_over_sampling(
        over_sample_size_lp=1
    )

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    source_galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_pix]
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array(
            [
                9.99914333e+01, 9.09781824e-02 ,7.07497204e-01, 8.49923287e-02,
                6.59363860e-01, 2.02882546e+00, 6.95897372e-01, 1.45637431e-01,
                6.64292575e-01, 1.08637423e-01
            ]
        ),
        1.0e-4,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-85.94918592874168, 1.0e-4)

    lens_galaxy_image = lens_galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.lp,
        psf=masked_dataset.psf,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image.array, 1.0e-2
    )
    assert fit_linear.model_images_of_planes_list[0] == pytest.approx(
        lens_galaxy_image.array, 1.0e-2
    )

    assert fit_linear.galaxy_model_image_dict[source_galaxy_pix][0] == pytest.approx(
        0.05425175, 1.0e-4
    )

    assert fit_linear.model_images_of_planes_list[1][0] == pytest.approx(
        0.05425175, 1.0e-4
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings_inversion=al.SettingsInversion(
            use_positive_only_solver=True,
        ),
        preloads=al.Preloads(
            mapper_indices=range(1, 10),
            source_pixel_zeroed_indices=np.array([1, 2, 3, 4, 6, 7, 8, 9])
        )
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
        abs=1.0e-1,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-86.01614801681916, 1.0e-4)


def test__simulate_imaging_data_and_fit__linear_light_profiles_and_pixelization__sub_2():

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, over_sample_size=2)

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
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.805
    )

    dataset = al.Imaging(
        data=dataset.data,
        psf=dataset.psf,
        noise_map=dataset.noise_map,
        over_sample_size_lp=2,
        over_sample_size_pixelization=2
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    lens_galaxy_linear = al.Galaxy(
        redshift=0.5,
        light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
        mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=0.01),
    )

    source_galaxy_pix = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer_linear = al.Tracer(
        galaxies=[lens_galaxy_linear, source_galaxy_pix]
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array(
            [
                99.97786229,  0.20198423, 1.60549092, 0.14487827,  1.6066755,   4.73452957,
                1.61532944,  0.32693971,  1.65517967,  0.19727977,
            ]
        ),
        1.0e-4,
    )

    assert fit_linear.figure_of_merit == pytest.approx(-86.288358299142, 1.0e-4)

    lens_galaxy_image = lens_galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.lp,
        psf=masked_dataset.psf,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[lens_galaxy_linear] == pytest.approx(
        lens_galaxy_image.array, 1.0e-2
    )
    assert fit_linear.model_images_of_planes_list[0] == pytest.approx(
        lens_galaxy_image.array, 1.0e-2
    )

    assert fit_linear.galaxy_model_image_dict[source_galaxy_pix][0] == pytest.approx(
        0.1330676, 1.0e-4
    )

    assert fit_linear.model_images_of_planes_list[1][0] == pytest.approx(
        0.1330676, 1.0e-4
    )

    assert fit_linear.subtracted_images_of_planes_list[1][0] == pytest.approx(
        0.34054169, 1.0e-4
    )

    fit_linear = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer_linear,
        settings_inversion=al.SettingsInversion(
            use_positive_only_solver=True,
            force_edge_pixels_to_zeros=True
        ),
        preloads=al.Preloads(
            mapper_indices=range(1, 10),
            source_pixel_zeroed_indices=np.array([1, 2, 3, 4, 6, 7, 8, 9])
        )
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
        abs=1.0e-1,
    )
    assert fit_linear.figure_of_merit == pytest.approx(-86.61380401245304, abs=1.0e-4)


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
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source_0 = al.Galaxy(redshift=0.4, pixelization=pixelization)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(3, 3)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    source_1 = al.Galaxy(redshift=0.5, pixelization=pixelization)

    tracer = al.Tracer(
        galaxies=[lens_0, lens_1, lens_2, source_0, source_1]
    )

    fit_mapping = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer,
    )

    masked_dataset_w_tilde = masked_dataset.apply_w_tilde()

    fit_w_tilde = al.FitImaging(
        dataset=masked_dataset_w_tilde,
        tracer=tracer,
    )

    assert fit_mapping.inversion.curvature_matrix == pytest.approx(
            fit_w_tilde.inversion.curvature_matrix,
        1.0e-4,
    )

    assert fit_mapping.inversion.regularization_matrix == pytest.approx(
            fit_w_tilde.inversion.regularization_matrix,
        1.0e-4,
    )


def test__fit_figure_of_merit__mge_mass_model(masked_imaging_7x7, masked_imaging_covariance_7x7):

    grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2,
                             over_sample_size=8)

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
        over_sample_size_lp=8
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.805
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

    masked_dataset = masked_dataset.apply_over_sampling(
        over_sample_size_lp=8
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