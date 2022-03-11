import os
from os import path

import pytest
from matplotlib import pyplot

from autofit import conf
from autolens import fixtures

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="config", autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "config"),
        output_path=path.join(directory, "pipeline", "output"),
    )
    return conf.instance


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    return plot_patch


@pytest.fixture(autouse=True, scope="session")
def remove_logs():
    yield
    for d, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".log"):
                os.remove(path.join(d, file))


############
# AutoLens #
############

# Lens Datasets #


@pytest.fixture(autouse=True)
def set_config_path(request):
    conf.instance.push(
        new_path=path.join(directory, "config"),
        output_path=path.join(directory, "output"),
    )


@pytest.fixture(name="mask_2d_7x7")
def make_mask_2d_7x7():
    return fixtures.make_mask_2d_7x7()


@pytest.fixture(name="sub_mask_2d_7x7")
def make_sub_mask_2d_7x7():
    return fixtures.make_sub_mask_2d_7x7()


@pytest.fixture(name="mask_2d_7x7_1_pix")
def make_mask_2d_7x7_1_pix():
    return fixtures.make_mask_2d_7x7_1_pix()


@pytest.fixture(name="blurring_mask_2d_7x7")
def make_blurring_mask_2d_7x7():
    return fixtures.make_blurring_mask_2d_7x7()


@pytest.fixture(name="grid_2d_7x7")
def make_grid_2d_7x7():
    return fixtures.make_grid_2d_7x7()


@pytest.fixture(name="sub_grid_2d_7x7")
def make_sub_grid_2d_7x7():
    return fixtures.make_sub_grid_2d_7x7()


@pytest.fixture(name="grid_2d_iterate_7x7")
def make_grid_2d_iterate_7x7():
    return fixtures.make_grid_2d_iterate_7x7()


@pytest.fixture(name="grid_2d_irregular_7x7")
def make_grid_2d_irregular_7x7():
    return fixtures.make_grid_2d_irregular_7x7()


@pytest.fixture(name="grid_2d_irregular_7x7_list")
def make_grid_2d_irregular_7x7_list():
    return fixtures.make_grid_2d_irregular_7x7_list()


@pytest.fixture(name="blurring_grid_2d_7x7")
def make_blurring_grid_2d_7x7(blurring_mask_2d_7x7):
    return fixtures.make_blurring_grid_2d_7x7()


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return fixtures.make_image_7x7()


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return fixtures.make_noise_map_7x7()


@pytest.fixture(name="psf_3x3")
def make_psf_3x3():
    return fixtures.make_psf_3x3()


@pytest.fixture(name="convolver_7x7")
def make_convolver_7x7():
    return fixtures.make_convolver_7x7()


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7():
    return fixtures.make_imaging_7x7()


@pytest.fixture(name="masked_imaging_7x7")
def make_masked_imaging_7x7():
    return fixtures.make_masked_imaging_7x7()


@pytest.fixture(name="masked_imaging_7x7_no_blur")
def make_masked_imaging_7x7_no_blur():
    return fixtures.make_masked_imaging_7x7_no_blur()


@pytest.fixture(name="visibilities_7")
def make_visibilities_7():
    return fixtures.make_visibilities_7()


@pytest.fixture(name="visibilities_noise_map_7")
def make_noise_map_7():
    return fixtures.make_visibilities_noise_map_7()


@pytest.fixture(name="uv_wavelengths_7x2")
def make_uv_wavelengths_7x2():
    return fixtures.make_uv_wavelengths_7x2()


@pytest.fixture(name="transformer_7x7_7")
def make_transformer_7x7_7():
    return fixtures.make_transformer_7x7_7()


@pytest.fixture(name="interferometer_7")
def make_interferometer_7():
    return fixtures.make_interferometer_7()


@pytest.fixture(name="interferometer_7_lop")
def make_interferometer_7_lop():
    return fixtures.make_interferometer_7_lop()


@pytest.fixture(name="interferometer_7_grid")
def make_interferometer_7_grid():
    return fixtures.make_interferometer_7_grid()


@pytest.fixture(name="positions_x2")
def make_positions_x2():
    return fixtures.make_positions_x2()


@pytest.fixture(name="positions_x2_noise_map")
def make_positions_x2_noise_map():
    return fixtures.make_positions_noise_map_x2()


@pytest.fixture(name="fluxes_x2")
def make_fluxes_x2():
    return fixtures.make_fluxes_x2()


@pytest.fixture(name="fluxes_x2_noise_map")
def make_fluxes_x2_noise_map():
    return fixtures.make_fluxes_noise_map_x2()


@pytest.fixture(name="point_dataset")
def make_point_dataset():
    return fixtures.make_point_dataset()


@pytest.fixture(name="point_dict")
def make_point_dict():
    return fixtures.make_point_dict()


# galaxies #


@pytest.fixture(name="ps_0")
def make_ps_0():
    return fixtures.make_ps_0()


@pytest.fixture(name="ps_1")
def make_ps_1():
    return fixtures.make_ps_1()


@pytest.fixture(name="lp_0")
def make_lp_0():
    return fixtures.make_lp_0()


@pytest.fixture(name="gal_x1_mp")
def make_gal_x1_mp():
    return fixtures.make_gal_x1_mp()


@pytest.fixture(name="gal_x1_lp")
def make_gal_x1_lp():
    return fixtures.make_gal_x1_lp()


# Ray Tracing #


@pytest.fixture(name="sub_grid_2d_7x7_simple")
def make_sub_grid_2d_7x7_simple():
    return fixtures.make_sub_grid_2d_7x7_simple()


@pytest.fixture(name="tracer_x1_plane_7x7")
def make_tracer_x1_plane_7x7(gal_x1_lp):
    return fixtures.make_tracer_x1_plane_7x7()


@pytest.fixture(name="tracer_x2_plane_7x7")
def make_tracer_x2_plane_7x7():
    return fixtures.make_tracer_x2_plane_7x7()


@pytest.fixture(name="tracer_x2_plane_inversion_7x7")
def make_tracer_x2_plane_inversion_7x7():
    return fixtures.make_tracer_x2_plane_inversion_7x7()


@pytest.fixture(name="tracer_x2_plane_voronoi_7x7")
def make_tracer_x2_plane_voronoi_7x7():
    return fixtures.make_tracer_x2_plane_voronoi_7x7()


@pytest.fixture(name="tracer_x2_plane_point")
def make_tracer_x2_plane_point():
    return fixtures.make_tracer_x2_plane_point()


### QUANTITY ###


@pytest.fixture(name="dataset_quantity_7x7_array_2d")
def make_dataset_quantity_7x7_array_2d():
    return fixtures.make_dataset_quantity_7x7_array_2d()


@pytest.fixture(name="fit_quantity_7x7_array_2d")
def make_fit_quantity_7x7_array_2d():
    return fixtures.make_fit_quantity_7x7_array_2d()


# Lens Fit #


@pytest.fixture(name="fit_imaging_x1_plane_7x7")
def make_fit_imaging_x1_plane_7x7():
    return fixtures.make_fit_imaging_x1_plane_7x7()


@pytest.fixture(name="fit_imaging_x2_plane_7x7")
def make_fit_imaging_x2_plane_7x7():
    return fixtures.make_fit_imaging_x2_plane_7x7()


@pytest.fixture(name="fit_imaging_x2_plane_inversion_7x7")
def make_fit_imaging_x2_plane_inversion_7x7():
    return fixtures.make_fit_imaging_x2_plane_inversion_7x7()


@pytest.fixture(name="fit_interferometer_x1_plane_7x7")
def make_fit_interferometer_x1_plane_7x7():
    return fixtures.make_fit_interferometer_x1_plane_7x7()


@pytest.fixture(name="fit_interferometer_x2_plane_7x7")
def make_fit_interferometer_x2_plane_7x7():
    return fixtures.make_fit_interferometer_x2_plane_7x7()


@pytest.fixture(name="fit_interferometer_x2_plane_inversion_7x7")
def make_fit_interferometer_x2_plane_inversion_7x7():
    return fixtures.make_fit_interferometer_x2_plane_inversion_7x7()


@pytest.fixture(name="fit_point_dataset_x2_plane")
def make_fit_point_dataset_x2_plane():
    return fixtures.make_fit_point_dataset_x2_plane()


@pytest.fixture(name="fit_point_dict_x2_plane")
def make_fit_point_dict_x2_plane():
    return fixtures.make_fit_point_dict_x2_plane()


### Analysis ###


@pytest.fixture(name="analysis_imaging_7x7")
def make_analysis_imaging_7x7():
    return fixtures.make_analysis_imaging_7x7()


@pytest.fixture(name="analysis_interferometer_7")
def make_analysis_interferometer_7():
    return fixtures.make_analysis_interferometer_7()


@pytest.fixture(name="analysis_point_x2")
def make_analysis_point_x2():
    return fixtures.make_analysis_point_x2()


@pytest.fixture(name="hyper_galaxy_image_0_7x7")
def make_hyper_galaxy_image_0_7x7():
    return fixtures.make_hyper_galaxy_image_0_7x7()


@pytest.fixture(name="hyper_model_image_7x7")
def make_hyper_model_image_7x7():
    return fixtures.make_hyper_model_image_7x7()


@pytest.fixture(name="hyper_galaxy_image_path_dict_7x7")
def make_hyper_galaxy_image_path_dict_7x7():
    return fixtures.make_hyper_galaxy_image_path_dict_7x7()


@pytest.fixture(name="include_2d_all")
def make_include_all():
    return fixtures.make_include_2d_all()


@pytest.fixture(name="samples_with_result")
def make_samples_with_result():
    return fixtures.make_samples_with_result()
