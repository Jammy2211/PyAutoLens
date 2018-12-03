# import os
# import shutil
#
# import pytest
#
# from autofit import conf
# from autolens.imaging import scaled_array
# from autolens.imaging import regular as im
# from autolens.imaging import masks as msk
# from autolens.profiles import light_profiles as lp
# from autolens.profiles import mass_profiles as mp
# from autolens.lensing import model_galaxy as g
# from autolens.inversion import pixelizations as pix
# from autolens.inversion import regularization as reg
# from autolens.inversion import inversions as inv
# from autolens.plotting import inversion_plotters
#
# import numpy as np
#
# @pytest.fixture(name='general_config')
# def test_general_config():
#     general_config_path = "{}/../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
#     conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")
#
#
# @pytest.fixture(name='inversion_plotter_path')
# def test_inversion_plotter_setup():
#     galaxy_plotter_path = "{}/../test_files/plotting/inversion/".format(os.path.dirname(os.path.realpath(__file__)))
#
#     if os.path.exists(galaxy_plotter_path):
#         shutil.rmtree(galaxy_plotter_path)
#
#     os.mkdir(galaxy_plotter_path)
#
#     return galaxy_plotter_path
#
# @pytest.fixture(name='regular')
# def test_image():
#
#     regular = scaled_array.ScaledSquarePixelArray(array=np.ones((3,3)), pixel_scale=1.0)
#     noise_map_ = im.NoiseMap(array=2.0*np.ones((3,3)), pixel_scale=1.0)
#     psf = im.PSF(array=3.0*np.ones((3,3)), pixel_scale=1.0)
#
#     return im.Image(array=regular, pixel_scale=1.0, noise_map_=noise_map_, psf=psf)
#
# @pytest.fixture(name='galaxy_light')
# def test_galaxy_light():
#     return g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))
#
#
# @pytest.fixture(name='galaxy_mass')
# def test_galaxy_mass():
#     return g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))
#
# @pytest.fixture(name='grids')
# def test_grids():
#     return msk.ImagingGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)
#
# @pytest.fixture(name='rectangular_pixelization')
# def test_rectangular_pixelization():
#     return pix.Rectangular(shape=(25, 25))
#
# @pytest.fixture(name='rectangular_mapper')
# def test_rectangular_mapper(rectangular_pixelization, grids):
#     return rectangular_pixelization.mapper_from_grids(grids=grids)
#
# @pytest.fixture(name='regularization')
# def test_regularization():
#     return reg.Constant(coeffs=(1.0,))
#
# @pytest.fixture(name='rectanguar_inversion')
# def test_rectangular_inversion(rectangular_mapper, regularization):
#     return inv.Inversion(im)
#
# def test__rectangular_inversion_is_output(rectangular_inversion, inversion_plotter_path):
#
#     inversion_plotters.plot_inversion(inversion=rectangular_inversion, should_plot_centres=True, should_plot_grid=True,
#                                 image_pixels=[[0, 1, 2], [3]], source_pixels=[[1, 2], [0]],
#                                 output_path=inversion_plotter_path, output_filename='rectangular_inversion',
#                                 output_format='png')
#     assert os.path.isfile(path=inversion_plotter_path + 'rectangular_inversion.png')
#     os.remove(path=inversion_plotter_path + 'rectangular_inversion.png')