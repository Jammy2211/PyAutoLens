from __future__ import division, print_function
import pytest
import numpy as np
from auto_lens.imaging import analysis_grids
from auto_lens.imaging import imaging
import os

test_data_dir = "{}/../../data/test_data/".format(os.path.dirname(os.path.realpath(__file__)))


class TestAnalysisGridCollection(object):
    
    
    class TestConstructor(object):
        
        def test__simple_grid_input__all_grids_used__sets_up_attributes(self):

            image_grid = analysis_grids.AnalysisGridImage(np.array([[1.0, 1.0],
                                                                   [2.0, 2.0],
                                                                   [3.0, 3.0]]))

            sub_grid = analysis_grids.AnalysisGridImageSub(np.array([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                                                                     [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]]),
                                                           sub_grid_size=2)

            blurring_grid = analysis_grids.AnalysisGridBlurring(np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], 
                                                                          [1.0, 1.0]]))
            
            ray_tracing_grids = analysis_grids.RayTracingGrids(image_grid, sub_grid, blurring_grid)
            
            assert (ray_tracing_grids.image.grid[0] == np.array([1.0, 1.0])).all()
            assert (ray_tracing_grids.image.grid[1] == np.array([2.0, 2.0])).all()
            assert (ray_tracing_grids.image.grid[2] == np.array([3.0, 3.0])).all()

            assert (ray_tracing_grids.sub.grid[0,0] == np.array([1.0, 1.0])).all()
            assert (ray_tracing_grids.sub.grid[0,1] == np.array([1.0, 1.0])).all()
            assert (ray_tracing_grids.sub.grid[0,2] == np.array([1.0, 1.0])).all()
            assert (ray_tracing_grids.sub.grid[0,3] == np.array([1.0, 1.0])).all()
            assert (ray_tracing_grids.sub.grid[1,0] == np.array([2.0, 2.0])).all()
            assert (ray_tracing_grids.sub.grid[1,1] == np.array([2.0, 2.0])).all()
            assert (ray_tracing_grids.sub.grid[1,2] == np.array([2.0, 2.0])).all()
            assert (ray_tracing_grids.sub.grid[1,3] == np.array([2.0, 2.0])).all()

            assert (ray_tracing_grids.blurring.grid[0] == np.array([1.0, 1.0])).all()
            assert (ray_tracing_grids.blurring.grid[0] == np.array([1.0, 1.0])).all()
            assert (ray_tracing_grids.blurring.grid[0] == np.array([1.0, 1.0])).all()
            assert (ray_tracing_grids.blurring.grid[0] == np.array([1.0, 1.0])).all()

        def test__simple_grid_input__sub_and_blurring_are_none__sets_up_attributes(self):

            image_grid = analysis_grids.AnalysisGridImage(np.array([[1.0, 1.0],
                                                                    [2.0, 2.0],
                                                                    [3.0, 3.0]]))

            ray_tracing_grids = analysis_grids.RayTracingGrids(image_grid)

            assert (ray_tracing_grids.image.grid[0] == np.array([1.0, 1.0])).all()
            assert (ray_tracing_grids.image.grid[1] == np.array([2.0, 2.0])).all()
            assert (ray_tracing_grids.image.grid[2] == np.array([3.0, 3.0])).all()

            assert ray_tracing_grids.sub == None

            assert ray_tracing_grids.blurring == None


    class TestFromMask(object):

        def test__all_grids_from_masks__correct_grids_setup(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_grid = mask.compute_image_grid()
            sub_grid = mask.compute_image_sub_grid(sub_grid_size=2)
            blurring_grid = mask.compute_blurring_grid(psf_size=(3,3))

            ray_tracing_grids = analysis_grids.RayTracingGrids.from_mask(mask, sub_grid_size=2, blurring_size=(3,3))

            assert (ray_tracing_grids.image.grid == image_grid).all()
            assert (ray_tracing_grids.sub.grid == sub_grid).all()
            assert (ray_tracing_grids.blurring.grid == blurring_grid).all()

        def test__sub_and_blurring_grids_are_none__correct_grids_setup(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_grid = mask.compute_image_grid()

            ray_tracing_grids = analysis_grids.RayTracingGrids.from_mask(mask)

            assert (ray_tracing_grids.image.grid == image_grid).all()
            assert ray_tracing_grids.sub == None
            assert ray_tracing_grids.blurring == None


class TestAnalysisGridImage(object):


    class TestConstructor:

        def test__simple_grid_input__sets_up_grid_in_attributes(self):

            grid = np.array([[1.0, 1.0],
                             [2.0, 2.0],
                             [3.0, 3.0]])

            analysis_grid = analysis_grids.AnalysisGridImage(grid)

            assert (analysis_grid.grid[0] == np.array([1.0, 1.0])).all()
            assert (analysis_grid.grid[1] == np.array([2.0, 2.0])).all()
            assert (analysis_grid.grid[2] == np.array([3.0, 3.0])).all()


    class TestFromMask:

        def test__simple_constructor__compare_to_manual_setup_via_mask(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_grid = mask.compute_image_grid()

            analysis_grid = analysis_grids.AnalysisGridImage(image_grid)

            analysis_grid_from_mask = analysis_grids.AnalysisGridImage.from_mask(mask)

            assert (analysis_grid.grid == analysis_grid_from_mask.grid).all()


class TestAnalysisGridImageSub(object):


    class TestConstructor:

        def test__simple_grid_input__sets_up_grid_in_attributes(self):

            grid = np.array([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                             [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]])

            analysis_grid = analysis_grids.AnalysisGridImageSub(grid=grid, sub_grid_size=2)

            assert (analysis_grid.grid[0,0] == np.array([1.0, 1.0])).all()
            assert (analysis_grid.grid[0,1] == np.array([1.0, 1.0])).all()
            assert (analysis_grid.grid[0,2] == np.array([1.0, 1.0])).all()
            assert (analysis_grid.grid[0,3] == np.array([1.0, 1.0])).all()
            assert (analysis_grid.grid[1,0] == np.array([2.0, 2.0])).all()
            assert (analysis_grid.grid[1,1] == np.array([2.0, 2.0])).all()
            assert (analysis_grid.grid[1,2] == np.array([2.0, 2.0])).all()
            assert (analysis_grid.grid[1,3] == np.array([2.0, 2.0])).all()


    class TestFromMask:

        def test__simple_constructor__compare_to_manual_setup_via_mask(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            image_sub_grid = mask.compute_image_sub_grid(sub_grid_size=2)

            analysis_grid = analysis_grids.AnalysisGridImageSub(image_sub_grid, sub_grid_size=2)

            analysis_grid_from_mask = analysis_grids.AnalysisGridImageSub.from_mask(mask, sub_grid_size=2)

            assert (analysis_grid.grid == analysis_grid_from_mask.grid).all()


class TestAnalysisGridBlurring(object):


    class TestConstructor:

        def test__simple_grid_input__sets_up_grid_in_attributes(self):

            grid = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

            analysis_grid = analysis_grids.AnalysisGridBlurring(grid=grid)

            assert (analysis_grid.grid[0] == np.array([1.0, 1.0])).all()
            assert (analysis_grid.grid[0] == np.array([1.0, 1.0])).all()
            assert (analysis_grid.grid[0] == np.array([1.0, 1.0])).all()
            assert (analysis_grid.grid[0] == np.array([1.0, 1.0])).all()


    class TestFromMask:

        def test__simple_constructor__compare_to_manual_setup_via_mask(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            blurring_grid = mask.compute_blurring_grid(psf_size=(3,3))

            analysis_grid = analysis_grids.AnalysisGridBlurring(blurring_grid)

            analysis_grid_from_mask = analysis_grids.AnalysisGridBlurring.from_mask(mask, psf_size=(3,3))

            assert (analysis_grid.grid == analysis_grid_from_mask.grid).all()


class TestAnalysisSparseMapper(object):

    class TestConstructor:

        def test__simple_mappeer_input__sets_up_grid_in_attributes(self):

            sparse_to_image = np.array([1, 2, 3, 5])
            image_to_sparse = np.array([6, 7, 2, 3])

            analysis_mapper = analysis_grids.AnalysisMapperSparse(sparse_to_image, image_to_sparse)

            assert (analysis_mapper.sparse_to_image == np.array([1, 2, 3, 5])).all()
            assert (analysis_mapper.image_to_sparse == np.array([6, 7, 2, 3])).all()

    class TestFromMask:

        def test__simple_constructor__compare_to_manual_setup_via_mask(self):

            mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])

            mask = imaging.Mask(mask=mask, pixel_scale=3.0)

            sparse_to_image, image_to_sparse = mask.compute_sparse_mappers(sparse_grid_size=1)

            analysis_mapper = analysis_grids.AnalysisMapperSparse(sparse_to_image, image_to_sparse)

            analysis_mapper_from_mask = analysis_grids.AnalysisMapperSparse.from_mask(mask, sparse_grid_size=1)

            assert (analysis_mapper.sparse_to_image == analysis_mapper_from_mask.sparse_to_image).all()