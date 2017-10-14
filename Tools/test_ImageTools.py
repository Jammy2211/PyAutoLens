from __future__ import division, print_function
import pytest
import numpy as np
from Tools import ImageTools

testdir = '/home/jammy/PycharmProjects/AutoLens/Data/testData/'

class TestLoadFits:
    
    def test__load_fits__input_fits_3x3_ones__loads_data_as_type_numpy_array(self):

        data2d, xy_dim = ImageTools.load_fits(workdir=testdir, file='3x3_ones.fits', hdu=0)

        assert type(data2d) == np.ndarray

    def test__load_fits__input_fits_3x3_ones__loads_correct_data(self):

        data2d, xy_dim = ImageTools.load_fits(workdir=testdir, file='3x3_ones.fits', hdu=0)

        assert (data2d == np.ones((3,3))).all()

    def test__load_fits__input_fits_4x3_ones__loads_correct_data(self):

        data2d, xy_dim = ImageTools.load_fits(workdir=testdir, file='4x3_ones.fits', hdu=0)

        assert (data2d == np.ones((4,3))).all()

    def test__load_fits__input_files_3x3_ones__loads_correct_dimensions(self):

        data2d, xy_dim = ImageTools.load_fits(workdir=testdir, file='3x3_ones.fits', hdu=0)

        assert xy_dim[0] == 3
        assert xy_dim[1] == 3

    def test__load_fits__input_files_4x3_ones__loads_correct_dimensions(self):

        data2d, xy_dim = ImageTools.load_fits(workdir=testdir, file='4x3_ones.fits', hdu=0)

        assert xy_dim[0] == 4
        assert xy_dim[1] == 3
        
class TestArcSec:

    def test__get_dimensions_arcsec__input_3x3_ones__calculates_correct_arcsecond_dimensions(self):

        xy_arcsec = ImageTools.get_dimensions_arcsec(xy_dim=[3,3], pixel_scale=0.1)

        assert xy_arcsec[0] == pytest.approx(0.3, 1e-5)
        assert xy_arcsec[1] == pytest.approx(0.3, 1e-5)

class TestGetMask:

    def test__get_mask_circular__odd_x_odd_mask_input_radius_1__correct_mask(self):

        mask2d = ImageTools.get_mask_circular(xy_dim=[3,3], pixel_scale=0.1, mask_radius_arcsec=0.08)

        assert (mask2d == np.array([[0,0,0],
                                    [0,1,0],
                                    [0,0,0]])).all()

    def test__get_mask_circular__odd_x_odd_mask_input_radius_2__correct_mask(self):

        mask2d = ImageTools.get_mask_circular(xy_dim=[3,3], pixel_scale=0.1, mask_radius_arcsec=0.1)

        assert (mask2d == np.array([[0,1,0],
                                    [1,1,1],
                                    [0,1,0]])).all()

    def test__get_mask_circular__odd_x_odd_mask_input_radius_3__correct_mask(self):

        mask2d = ImageTools.get_mask_circular(xy_dim=[3,3], pixel_scale=0.1, mask_radius_arcsec=0.3)

        assert (mask2d == np.array([[1,1,1],
                                    [1,1,1],
                                    [1,1,1]])).all()

    def test__get_mask_circular__even_x_odd_mask_input_radius_1__correct_mask(self):

        mask2d = ImageTools.get_mask_circular(xy_dim=[4,3], pixel_scale=0.1, mask_radius_arcsec=0.05)

        assert (mask2d == np.array([[0,0,0],
                                    [0,1,0],
                                    [0,1,0],
                                    [0,0,0]])).all()

    def test__get_mask_circular__even_x_odd_mask_input_radius_2__correct_mask(self):

        mask2d = ImageTools.get_mask_circular(xy_dim=[4,3], pixel_scale=0.1, mask_radius_arcsec=0.150001)

        assert (mask2d == np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [1, 1, 1],
                                    [0, 1, 0]])).all()

    def test__get_mask_circular__even_x_odd_mask_input_radius_3__correct_mask(self):

        mask2d = ImageTools.get_mask_circular(xy_dim=[4,3], pixel_scale=0.1, mask_radius_arcsec=0.3)

        assert (mask2d == np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])).all()

    def test__get_mask_circular__even_x_even_mask_input_radius_1__correct_mask(self):

        mask2d = ImageTools.get_mask_circular(xy_dim=[4,4], pixel_scale=0.1, mask_radius_arcsec=0.072)

        assert (mask2d == np.array([[0,0,0,0],
                                    [0,1,1,0],
                                    [0,1,1,0],
                                    [0,0,0,0]])).all()

    def test__get_mask_circular__even_x_even_mask_input_radius_2__correct_mask(self):

        mask2d = ImageTools.get_mask_circular(xy_dim=[4,4], pixel_scale=0.1, mask_radius_arcsec=0.17)

        assert (mask2d == np.array([[0,1,1,0],
                                    [1,1,1,1],
                                    [1,1,1,1],
                                    [0,1,1,0]])).all()

    def test__get_mask_circular__even_x_even_mask_input_radius_3__correct_mask(self):

        mask2d = ImageTools.get_mask_circular(xy_dim=[4,4], pixel_scale=0.1, mask_radius_arcsec=0.3)

        assert (mask2d == np.array([[1,1,1,1],
                                    [1,1,1,1],
                                    [1,1,1,1],
                                    [1,1,1,1]])).all()

class TestSkyBg:

    def test__estimate_sky_level_via_edges__input_all_ones__sky_bg_level_1(self):

        mean, sigma = ImageTools.estimate_sky_via_edges(image=np.ones((3, 3)))

        assert mean == 1.0
        assert sigma == 0.0
        
    def test__estimate_sky_level_via_edges__3x3_image_simple_gaussian__answer_ignores_central_pixel(self):

        image = np.array([[1,1,1],
                          [1,100,1],
                          [1,1,1]])

        mean, sigma = ImageTools.estimate_sky_via_edges(image=image)

        assert mean == 1.0
        assert sigma == 0.0
        
    def test__estimate_sky_level_via_edges__4x3_image_simple_gaussian__ignores_central_pixels(self):

        image = np.array([[1,1,1],
                          [1,100,1],
                          [1,100,1],
                          [1,1,1]])

        mean, sigma = ImageTools.estimate_sky_via_edges(image=image)

        assert mean == 1.0
        assert sigma == 0.0

    def test__estimate_sky_level_via_edges__4x4_image_simple_gaussian__ignores_central_pixels(self):

        image = np.array([[1, 1,  1, 1],
                          [1,100,100,1],
                          [1,100,100,1],
                          [1, 1, 1, 1]])

        mean, sigma = ImageTools.estimate_sky_via_edges(image=image)

        assert mean == 1.0
        assert sigma == 0.0

    def test__estimate_sky_level_via_edges__3x3_image__correct_values(self):

        image = np.array([[0, 1,  2, 3],
                          [4,100,100,5],
                          [6,100,100,7],
                          [8, 9, 10, 11]])

        mean, sigma = ImageTools.estimate_sky_via_edges(image=image)

        assert mean == np.mean(np.arange(12))
        assert sigma == np.std(np.arange(12))

    def test__estimate_sky_level_via_edges__5x5_image_simple_gaussian_two_edges__ignores_central_pixel(self):

        image = np.array([[1,1, 1, 1,1],
                          [1,1, 1, 1,1],
                          [1,1,100,1,1],
                          [1,1, 1, 1,1],
                          [1,1, 1, 1,1]])

        mean, sigma = ImageTools.estimate_sky_via_edges(image=image, no_edges=2)

        assert mean == 1.0
        assert sigma == 0.0

    def test__estimate_sky_level_via_edges__6x5_image_two_edges__correct_values(self):

        image = np.array([[0,  1,  2,   3 , 4],
                          [5,  6,  7,   8 , 9],
                          [10, 11, 100, 12 ,13],
                          [14, 15, 100, 16, 17],
                          [18, 19,  20, 21, 22],
                          [23, 24,  25, 26, 27]])

        mean, sigma = ImageTools.estimate_sky_via_edges(image=image, no_edges=2)

        assert mean == np.mean(np.arange(28))
        assert sigma == np.std(np.arange(28))

    def test__estimate_sky_level_via_edges__7x7_image_three_edges__correct_values(self):

        image = np.array([[0,  1,  2,  3 ,  4,  5, 6],
                          [7,  8,  9,  10 , 11, 12, 13],
                          [14, 15, 16, 17  ,18, 19, 20],
                          [21, 22, 23, 100, 24, 25, 26],
                          [27, 28, 29, 30,  31, 32, 33],
                          [34, 35, 36, 37,  38, 39, 40],
                          [41, 42, 43, 44,  45, 46, 47]])

        mean, sigma = ImageTools.estimate_sky_via_edges(image=image, no_edges=3)

        assert mean == np.mean(np.arange(48))
        assert sigma == np.std(np.arange(48))