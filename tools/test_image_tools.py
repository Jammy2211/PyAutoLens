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

