#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
File: tests/python/CTI_Tools_test.py

Created on: 06/02/17
Author: user
"""

from __future__ import division, print_function

import pytest
from LensData.ArrayTools import *
import numpy as np

#@pytest.fixture(scope='function')
#def lh_match():
#    lh_match.data = np.zeros((2, 2))
#    return lh_match

class TestArrayTools:

    def test_get_dimensions_pixels_SquareArray_CorrectXYDims(self):

        xdim, ydim = get_dimensions_pixels(np.zeros((2,2)))

        assert xdim == 2
        assert ydim == 2

    def test_get_dimensions_pixels_RectangleArray1_CorrectXYDims(self):

        xdim, ydim = get_dimensions_pixels(np.zeros((2, 4)))

        assert xdim == 2
        assert ydim == 4

    def test_get_dimensions_pixels_RectangleArray2_CorrectXYDims(self):

        xdim, ydim = get_dimensions_pixels(np.zeros((3, 1)))

        assert xdim == 3
        assert ydim == 1

    def test_get_dimensions_pixels_Not2DArray_RaiseIndexError(self):

        with pytest.raises(IndexError):
            get_dimensions_pixels(np.zeros((2)))

        with pytest.raises(IndexError):
            get_dimensions_pixels(np.zeros((2,2,2)))