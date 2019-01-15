import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens.data.array import grids, mask
from autolens.model.inversion import pixelizations, regularization
from autolens.model.galaxy import galaxy as g
from autolens.lens.util import plane_util, ray_tracing_util
from autolens.lens.stack import plane_stack
from autolens.lens.stack import ray_tracing_stack
from autolens.lens import ray_tracing
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp

@pytest.fixture(name="grid_stack_0")
def make_grid_stack_0():
    ma = mask.Mask(np.array([[True, True, True, True],
                             [True, False, False, True],
                             [True, True, True, True]]), pixel_scale=6.0)

    grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=2,
                                                                                       psf_shape=(3, 3))

    # Manually overwrite a set of cooridnates to make tests of grid_stacks and defledctions straightforward

    grid_stack_0.regular[0] = np.array([1.0, 1.0])
    grid_stack_0.regular[1] = np.array([1.0, 0.0])
    grid_stack_0.sub[0] = np.array([1.0, 1.0])
    grid_stack_0.sub[1] = np.array([1.0, 0.0])
    grid_stack_0.sub[2] = np.array([1.0, 1.0])
    grid_stack_0.sub[3] = np.array([1.0, 0.0])
    grid_stack_0.blurring[0] = np.array([1.0, 0.0])

    return grid_stack_0


@pytest.fixture(name="grid_stack_1")
def make_grid_stack_1():
    ma = mask.Mask(np.array([[True, True, True, True],
                             [True, False, False, True],
                             [True, True, True, True]]), pixel_scale=12.0)

    grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=2,
                                                                                       psf_shape=(3, 3))

    # Manually overwrite a set of cooridnates to make tests of grid_stacks and defledctions straightforward

    grid_stack_0.regular[0] = np.array([2.0, 2.0])
    grid_stack_0.regular[1] = np.array([2.0, 0.0])
    grid_stack_0.sub[0] = np.array([2.0, 2.0])
    grid_stack_0.sub[1] = np.array([2.0, 0.0])
    grid_stack_0.sub[2] = np.array([2.0, 2.0])
    grid_stack_0.sub[3] = np.array([2.0, 0.0])
    grid_stack_0.blurring[0] = np.array([2.0, 0.0])

    return grid_stack_0


@pytest.fixture(name="padded_grid_stack")
def make_padded_grid_stack():
    ma = mask.Mask(np.array([[True, False]]), pixel_scale=3.0)
    return grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(ma, 2, (3, 3))


@pytest.fixture(name='galaxy_non', scope='function')
def make_galaxy_non():
    return g.Galaxy()


@pytest.fixture(name="galaxy_light")
def make_galaxy_light():
    return g.Galaxy(light_profile=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                      effective_radius=0.6, sersic_index=4.0))


@pytest.fixture(name="galaxy_mass")
def make_galaxy_mass():
    return g.Galaxy(mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))


@pytest.fixture(name='galaxy_mass_x2')
def make_galaxy_mass_x2():
    return g.Galaxy(sis_0=mp.SphericalIsothermal(einstein_radius=1.0),
                    sis_1=mp.SphericalIsothermal(einstein_radius=1.0))


class TestTracerImagePlaneStack(object):

    class TestImagePlaneImage:

        def test__1_plane_returns_same_as_tracer__x2_grids__returns_x2_images(self, grid_stack_0, grid_stack_1):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))

            image_plane = plane_stack.PlaneStack(galaxies=[g0, g1, g2], grid_stacks=[grid_stack_0, grid_stack_1],
                                   compute_deflections=True)

            tracer = ray_tracing_stack.TracerImagePlaneStack(lens_galaxies=[g0, g1, g2],
                                                       image_plane_grid_stacks=[grid_stack_0, grid_stack_1])

            assert (tracer.image_plane_images_1d[0] == image_plane.image_plane_images_1d[0]).all()

            image_plane_image_2d = \
                grid_stack_0.regular.scaled_array_from_array_1d(image_plane.image_plane_images_1d[0])
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[0]).all()

            assert (tracer.image_plane_images_1d[1] == image_plane.image_plane_images_1d[1]).all()

            image_plane_image_2d = \
                grid_stack_0.regular.scaled_array_from_array_1d(image_plane.image_plane_images_1d[1])
            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[1]).all()

    class TestImagePlaneBlurringImages:

        def test__1_plane_returns_same_as_tracer__x2_grids__returns_x2_images(self, grid_stack_0, grid_stack_1):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))

            image_plane = plane_stack.PlaneStack(galaxies=[g0, g1, g2], grid_stacks=[grid_stack_0, grid_stack_1],
                                   compute_deflections=True)

            tracer = ray_tracing_stack.TracerImagePlaneStack(lens_galaxies=[g0, g1, g2],
                                                       image_plane_grid_stacks=[grid_stack_0, grid_stack_1])

            assert (tracer.image_plane_blurring_images_1d[0] == image_plane.image_plane_blurring_images_1d[0]).all()
            assert (tracer.image_plane_blurring_images_1d[1] == image_plane.image_plane_blurring_images_1d[1]).all()

    class TestCompareToNonStacks:

        def test__compare_all_quantities_to_non_stack_tracers(self, grid_stack_0, grid_stack_1):

            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))


            tracer_stack = ray_tracing_stack.TracerImagePlaneStack(lens_galaxies=[g0, g1, g2],
                                                       image_plane_grid_stacks=[grid_stack_0, grid_stack_1])

            tracer_0 = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack_0)

            assert (tracer_stack.image_plane_images[0] == tracer_0.image_plane_image).all()
            assert (tracer_stack.image_plane_blurring_images_1d[0] == tracer_0.image_plane_blurring_image_1d).all()

            tracer_1 = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1, g2], image_plane_grid_stack=grid_stack_1)

            assert (tracer_stack.image_plane_images[1] == tracer_1.image_plane_image).all()
            assert (tracer_stack.image_plane_blurring_images_1d[1] == tracer_1.image_plane_blurring_image_1d).all()


class TestTracerImageSourcePlanesStack(object):
    class TestSetup:

        def test__x2_grid_stack__no_galaxy__image_and_source_planes_setup__same_coordinates(self, grid_stack_0,
                                                                                            grid_stack_1, galaxy_non):
            tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[galaxy_non], source_galaxies=[galaxy_non],
                                                              image_plane_grid_stacks=[grid_stack_0, grid_stack_1])

            assert tracer.image_plane.grid_stacks[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stacks[0].sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stacks[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.image_plane.grid_stacks[0].sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.image_plane.grid_stacks[0].sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflection_stacks[0].regular[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflection_stacks[0].sub[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflection_stacks[0].sub[1] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflection_stacks[0].sub[2] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflection_stacks[0].sub[3] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflection_stacks[0].blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

            assert tracer.source_plane.grid_stacks[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grid_stacks[0].sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grid_stacks[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-3)
            assert tracer.source_plane.grid_stacks[0].sub[2] == pytest.approx(np.array([1.0, 1.0]), 1e-3)
            assert tracer.source_plane.grid_stacks[0].sub[3] == pytest.approx(np.array([1.0, 0.0]), 1e-3)

            assert tracer.image_plane.grid_stacks[1].regular[0] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.image_plane.grid_stacks[1].sub[0] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.image_plane.grid_stacks[1].sub[1] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
            assert tracer.image_plane.grid_stacks[1].sub[2] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.image_plane.grid_stacks[1].sub[3] == pytest.approx(np.array([2.0, 0.0]), 1e-3)

            assert tracer.image_plane.deflection_stacks[1].regular[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflection_stacks[1].sub[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflection_stacks[1].sub[1] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflection_stacks[1].sub[2] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflection_stacks[1].sub[3] == pytest.approx(np.array([0.0, 0.0]), 1e-3)
            assert tracer.image_plane.deflection_stacks[1].blurring[0] == pytest.approx(np.array([0.0, 0.0]), 1e-3)

            assert tracer.source_plane.grid_stacks[1].regular[0] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.source_plane.grid_stacks[1].sub[0] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.source_plane.grid_stacks[1].sub[1] == pytest.approx(np.array([2.0, 0.0]), 1e-3)
            assert tracer.source_plane.grid_stacks[1].sub[2] == pytest.approx(np.array([2.0, 2.0]), 1e-3)
            assert tracer.source_plane.grid_stacks[1].sub[3] == pytest.approx(np.array([2.0, 0.0]), 1e-3)

    class TestImagePlaneImages:

        def test__same_as_above__x2_grids__returns_x2_images(self, grid_stack_0, grid_stack_1):
            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0),
                          mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            image_plane = plane_stack.PlaneStack(galaxies=[g0], grid_stacks=[grid_stack_0, grid_stack_1],
                                        compute_deflections=True)
            source_plane_grid_stacks = image_plane.trace_grids_to_next_plane()
            source_plane = plane_stack.PlaneStack(galaxies=[g0], grid_stacks=source_plane_grid_stacks, compute_deflections=False)

            tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[g0], source_galaxies=[g0],
                                                              image_plane_grid_stacks=[grid_stack_0, grid_stack_1])

            assert (tracer.image_plane_images_1d[0] == image_plane.image_plane_images_1d[0] +
                    source_plane.image_plane_images_1d[0]).all()

            image_plane_image_2d = \
                grid_stack_0.regular.scaled_array_from_array_1d(image_plane.image_plane_images_1d[0]) + \
                grid_stack_0.regular.scaled_array_from_array_1d(source_plane.image_plane_images_1d[0])

            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[0]).all()

            assert (tracer.image_plane_images_1d[1] == image_plane.image_plane_images_1d[1] +
                    source_plane.image_plane_images_1d[1]).all()

            image_plane_image_2d = \
                grid_stack_0.regular.scaled_array_from_array_1d(image_plane.image_plane_images_1d[1]) + \
                grid_stack_0.regular.scaled_array_from_array_1d(source_plane.image_plane_images_1d[1])

            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images[1]).all()

        def test__padded_2d_image_from_plane__mapped_correctly(self, padded_grid_stack, galaxy_light, galaxy_mass):
            tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[galaxy_light, galaxy_mass],
                                                              source_galaxies=[galaxy_light],
                                                              image_plane_grid_stacks=[padded_grid_stack,
                                                                                       padded_grid_stack])

            image_plane_image_2d = \
                padded_grid_stack.regular.scaled_array_from_array_1d(tracer.image_plane.image_plane_images_1d[0]) + \
                padded_grid_stack.regular.scaled_array_from_array_1d(tracer.source_plane.image_plane_images_1d[0])

            assert image_plane_image_2d.shape == (1, 2)
            assert (image_plane_image_2d == tracer.image_plane_images[0]).all()

            image_plane_image_2d = \
                padded_grid_stack.regular.scaled_array_from_array_1d(tracer.image_plane.image_plane_images_1d[1]) + \
                padded_grid_stack.regular.scaled_array_from_array_1d(tracer.source_plane.image_plane_images_1d[1])

            assert (image_plane_image_2d == tracer.image_plane_images[1]).all()

        def test__padded_2d_image_for_simulation__mapped_correctly_not_trimmed(self, padded_grid_stack, galaxy_light,
                                                                               galaxy_mass):
            tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[galaxy_light, galaxy_mass],
                                                              source_galaxies=[galaxy_light],
                                                              image_plane_grid_stacks=[padded_grid_stack,
                                                                                       padded_grid_stack])

            image_plane_image_2d = \
                padded_grid_stack.regular.map_to_2d_keep_padded(tracer.image_plane.image_plane_images_1d[0]) + \
                padded_grid_stack.regular.map_to_2d_keep_padded(tracer.source_plane.image_plane_images_1d[0])

            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images_for_simulation[0]).all()

            image_plane_image_2d = \
                padded_grid_stack.regular.map_to_2d_keep_padded(tracer.image_plane.image_plane_images_1d[1]) + \
                padded_grid_stack.regular.map_to_2d_keep_padded(tracer.source_plane.image_plane_images_1d[1])

            assert image_plane_image_2d.shape == (3, 4)
            assert (image_plane_image_2d == tracer.image_plane_images_for_simulation[1]).all()

    class TestImagePlaneBlurringImages:

        def test__galaxy_with_liht_and_mass__x2_grids_in__x2_images_out(self, grid_stack_0, grid_stack_1):
            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0),
                          mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

            image_plane = plane_stack.PlaneStack(galaxies=[g0], grid_stacks=[grid_stack_0, grid_stack_1],
                                        compute_deflections=True)

            deflections_grid_0 = plane_util.deflections_of_galaxies_from_grid_stack(grid_stack=grid_stack_0, galaxies=[g0])
            source_grid_0 = ray_tracing_util.traced_collection_for_deflections(grid_stack=grid_stack_0,
                                                                 deflections=deflections_grid_0)

            deflections_grid_1 = plane_util.deflections_of_galaxies_from_grid_stack(grid_stack=grid_stack_1, galaxies=[g0])
            source_grid_1 = ray_tracing_util.traced_collection_for_deflections(grid_stack=grid_stack_1,
                                                                 deflections=deflections_grid_1)

            source_plane = plane_stack.PlaneStack(galaxies=[g1], grid_stacks=[source_grid_0, source_grid_1],
                                         compute_deflections=False)

            tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[g0], source_galaxies=[g1],
                                                              image_plane_grid_stacks=[grid_stack_0, grid_stack_1])

            image_plane_blurring_image = image_plane.image_plane_blurring_images_1d[0] + \
                                         source_plane.image_plane_blurring_images_1d[0]

            assert (image_plane_blurring_image == tracer.image_plane_blurring_images_1d[0]).all()

            image_plane_blurring_image = image_plane.image_plane_blurring_images_1d[1] + \
                                         source_plane.image_plane_blurring_images_1d[1]

            assert (image_plane_blurring_image == tracer.image_plane_blurring_images_1d[1]).all()

    class TestImagePlanePixGrid:

        def test__galaxies_have_no_pixelization__no_pix_grid_added(self):
            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]]), pixel_scale=1.0)

            grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                            psf_shape=(1, 1))

            galaxy = g.Galaxy()

            tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                              image_plane_grid_stacks=[grid_stack_0])

            assert (tracer.image_plane.grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.source_plane.grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxies_have_other_pixelization__returns_normal_grid_stacks(self):
            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]]), pixel_scale=1.0)

            grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                            psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                              image_plane_grid_stacks=[grid_stack_0])

            assert (tracer.image_plane.grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.source_plane.grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__but_grid_is_padded_grid__returns_normal_grid_stacks(
                self):
            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, True, False]]), pixel_scale=1.0)

            grid_stack_0 = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma,
                                                                                                   sub_grid_size=1,
                                                                                                   psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                              image_plane_grid_stacks=[grid_stack_0])

            assert (tracer.image_plane.grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.source_plane.grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__returns_grid_stacks_with_pix_grid(self):
            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, True, False]]), pixel_scale=1.0)

            grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                            psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant())

            tracer = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[galaxy], source_galaxies=[galaxy],
                                                              image_plane_grid_stacks=[grid_stack_0])

            assert (tracer.image_plane.grid_stacks[0].regular == grid_stack_0.regular).all()
            assert (tracer.image_plane.grid_stacks[0].sub == grid_stack_0.sub).all()
            assert (tracer.image_plane.grid_stacks[0].blurring == grid_stack_0.blurring).all()
            assert (tracer.image_plane.grid_stacks[0].pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                                       [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                       [-1.0, -1.0], [-1.0, 1.0]])).all()

            assert (tracer.source_plane.grid_stacks[0].regular == grid_stack_0.regular).all()
            assert (tracer.source_plane.grid_stacks[0].sub == grid_stack_0.sub).all()
            assert (tracer.source_plane.grid_stacks[0].blurring == grid_stack_0.blurring).all()
            assert (tracer.source_plane.grid_stacks[0].pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                                        [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                        [-1.0, -1.0], [-1.0, 1.0]])).all()

    class TestCompareToNonStacks:

        def test__compare_all_quantities_to_non_stack_tracers(self, grid_stack_0, grid_stack_1):
            g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
            g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))
            g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0))

            tracer_stack = ray_tracing_stack.TracerImageSourcePlanesStack(lens_galaxies=[g0, g1, g2],
                                                                    source_galaxies=[g0, g1],
                                                                    image_plane_grid_stacks=[grid_stack_0,
                                                                                             grid_stack_1])

            tracer_0 = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1, g2], source_galaxies=[g0, g1],
                                                           image_plane_grid_stack=grid_stack_0)

            assert (tracer_stack.image_plane_images[0] == tracer_0.image_plane_image).all()
            assert (tracer_stack.image_plane_blurring_images_1d[0] == tracer_0.image_plane_blurring_image_1d).all()

            tracer_1 = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1, g2], source_galaxies=[g0, g1],
                                                           image_plane_grid_stack=grid_stack_1)

            assert (tracer_stack.image_plane_images[1] == tracer_1.image_plane_image).all()
            assert (tracer_stack.image_plane_blurring_images_1d[1] == tracer_1.image_plane_blurring_image_1d).all()


class TestMultiTracerStack(object):
    class TestPlaneGridStacks:

        def test__4_planes__data_grid_and_deflection_stacks_are_correct__sis_mass_profile(self, grid_stack_0):
            g0 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g1 = g.Galaxy(redshift=2.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g2 = g.Galaxy(redshift=0.1, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g3 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g4 = g.Galaxy(redshift=1.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))
            g5 = g.Galaxy(redshift=3.0, mass_profile=mp.SphericalIsothermal(einstein_radius=1.0))

            tracer = ray_tracing_stack.TracerMultiPlanesStack(galaxies=[g0, g1, g2, g3, g4, g5],
                                                        image_plane_grid_stacks=[grid_stack_0, grid_stack_0],
                                                        cosmology=cosmo.Planck15)

            # From unit test below:
            # Beta_01 = 0.9348
            # Beta_02 = 0.9840
            # Beta_03 = 1.0
            # Beta_12 = 0.754
            # Beta_13 = 1.0
            # Beta_23 = 1.0

            val = np.sqrt(2) / 2.0

            assert tracer.planes[0].grid_stacks[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
            assert tracer.planes[0].grid_stacks[0].sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
            assert tracer.planes[0].grid_stacks[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)
            assert tracer.planes[0].grid_stacks[0].blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

            assert tracer.planes[0].deflection_stacks[0].regular[0] == pytest.approx(np.array([val, val]), 1e-4)
            assert tracer.planes[0].deflection_stacks[0].sub[0] == pytest.approx(np.array([val, val]), 1e-4)
            assert tracer.planes[0].deflection_stacks[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)
            assert tracer.planes[0].deflection_stacks[0].blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

            assert tracer.planes[1].grid_stacks[0].regular[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]), 1e-4)
            assert tracer.planes[1].grid_stacks[0].sub[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]), 1e-4)
            assert tracer.planes[1].grid_stacks[0].sub[1] == pytest.approx(
                np.array([(1.0 - 0.9348 * 1.0), 0.0]), 1e-4)
            assert tracer.planes[1].grid_stacks[0].blurring[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * 1.0), 0.0]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]))
            defl12 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * 1.0), 0.0]]))

            assert tracer.planes[1].deflection_stacks[0].regular[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflection_stacks[0].sub[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflection_stacks[0].sub[1] == pytest.approx(defl12[0], 1e-4)
            assert tracer.planes[1].deflection_stacks[0].blurring[0] == pytest.approx(defl12[0], 1e-4)

            assert tracer.planes[2].grid_stacks[0].regular[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                          (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grid_stacks[0].sub[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                          (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grid_stacks[0].sub[1] == pytest.approx(
                np.array([(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]),
                          0.0]), 1e-4)
            assert tracer.planes[2].grid_stacks[0].blurring[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]),
                          0.0]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                                                                    (1.0 - 0.9839601 * val - 0.7539734 * defl11[
                                                                        0, 1])]]))
            defl22 = 2.0 * g0.deflections_from_grid(
                grid=np.array([[(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]), 0.0]]))

            assert tracer.planes[2].deflection_stacks[0].regular[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflection_stacks[0].sub[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflection_stacks[0].sub[1] == pytest.approx(defl22[0], 1e-4)
            assert tracer.planes[2].deflection_stacks[0].blurring[0] == pytest.approx(defl22[0], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflection_stacks[0].regular[0, 0] -
                      tracer.planes[1].deflection_stacks[0].regular[0, 0] -
                      tracer.planes[2].deflection_stacks[0].regular[0, 0])

            coord2 = (1.0 - tracer.planes[0].deflection_stacks[0].regular[0, 1] -
                      tracer.planes[1].deflection_stacks[0].regular[0, 1] -
                      tracer.planes[2].deflection_stacks[0].regular[0, 1])

            coord3 = (1.0 - tracer.planes[0].deflection_stacks[0].sub[1, 0] -
                      tracer.planes[1].deflection_stacks[0].sub[1, 0] -
                      tracer.planes[2].deflection_stacks[0].sub[1, 0])

            assert tracer.planes[3].grid_stacks[0].regular[0] == pytest.approx(np.array([coord1, coord2]), 1e-4)
            assert tracer.planes[3].grid_stacks[0].sub[0] == pytest.approx(np.array([coord1, coord2]), 1e-4)
            assert tracer.planes[3].grid_stacks[0].sub[1] == pytest.approx(np.array([coord3, 0.0]), 1e-4)
            assert tracer.planes[3].grid_stacks[0].blurring[0] == pytest.approx(np.array([coord3, 0.0]), 1e-4)

            assert tracer.planes[0].grid_stacks[0].regular[0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
            assert tracer.planes[0].grid_stacks[0].sub[0] == pytest.approx(np.array([1.0, 1.0]), 1e-4)
            assert tracer.planes[0].grid_stacks[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)
            assert tracer.planes[0].grid_stacks[0].blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

            assert tracer.planes[0].deflection_stacks[0].regular[0] == pytest.approx(np.array([val, val]), 1e-4)
            assert tracer.planes[0].deflection_stacks[0].sub[0] == pytest.approx(np.array([val, val]), 1e-4)
            assert tracer.planes[0].deflection_stacks[0].sub[1] == pytest.approx(np.array([1.0, 0.0]), 1e-4)
            assert tracer.planes[0].deflection_stacks[0].blurring[0] == pytest.approx(np.array([1.0, 0.0]), 1e-4)

            assert tracer.planes[1].grid_stacks[0].regular[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]), 1e-4)
            assert tracer.planes[1].grid_stacks[0].sub[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]), 1e-4)
            assert tracer.planes[1].grid_stacks[0].sub[1] == pytest.approx(
                np.array([(1.0 - 0.9348 * 1.0), 0.0]), 1e-4)
            assert tracer.planes[1].grid_stacks[0].blurring[0] == pytest.approx(
                np.array([(1.0 - 0.9348 * 1.0), 0.0]), 1e-4)

            defl11 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * val), (1.0 - 0.9348 * val)]]))
            defl12 = g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9348 * 1.0), 0.0]]))

            assert tracer.planes[1].deflection_stacks[0].regular[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflection_stacks[0].sub[0] == pytest.approx(defl11[0], 1e-4)
            assert tracer.planes[1].deflection_stacks[0].sub[1] == pytest.approx(defl12[0], 1e-4)
            assert tracer.planes[1].deflection_stacks[0].blurring[0] == pytest.approx(defl12[0], 1e-4)

            assert tracer.planes[2].grid_stacks[0].regular[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                          (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grid_stacks[0].sub[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                          (1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 1])]), 1e-4)
            assert tracer.planes[2].grid_stacks[0].sub[1] == pytest.approx(
                np.array([(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]),
                          0.0]), 1e-4)
            assert tracer.planes[2].grid_stacks[0].blurring[0] == pytest.approx(
                np.array([(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]),
                          0.0]), 1e-4)

            # 2 Galaxies in this plane, so multiply by 2.0

            defl21 = 2.0 * g0.deflections_from_grid(grid=np.array([[(1.0 - 0.9839601 * val - 0.7539734 * defl11[0, 0]),
                                                                    (1.0 - 0.9839601 * val - 0.7539734 * defl11[
                                                                        0, 1])]]))
            defl22 = 2.0 * g0.deflections_from_grid(
                grid=np.array([[(1.0 - 0.9839601 * 1.0 - 0.7539734 * defl12[0, 0]), 0.0]]))

            assert tracer.planes[2].deflection_stacks[1].regular[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflection_stacks[1].sub[0] == pytest.approx(defl21[0], 1e-4)
            assert tracer.planes[2].deflection_stacks[1].sub[1] == pytest.approx(defl22[0], 1e-4)
            assert tracer.planes[2].deflection_stacks[1].blurring[0] == pytest.approx(defl22[0], 1e-4)

            coord1 = (1.0 - tracer.planes[0].deflection_stacks[1].regular[0, 0] -
                      tracer.planes[1].deflection_stacks[1].regular[0, 0] -
                      tracer.planes[2].deflection_stacks[1].regular[0, 0])

            coord2 = (1.0 - tracer.planes[0].deflection_stacks[1].regular[0, 1] -
                      tracer.planes[1].deflection_stacks[1].regular[0, 1] -
                      tracer.planes[2].deflection_stacks[1].regular[0, 1])

            coord3 = (1.0 - tracer.planes[0].deflection_stacks[1].sub[1, 0] -
                      tracer.planes[1].deflection_stacks[1].sub[1, 0] -
                      tracer.planes[2].deflection_stacks[1].sub[1, 0])

            assert tracer.planes[3].grid_stacks[1].regular[0] == pytest.approx(np.array([coord1, coord2]), 1e-4)
            assert tracer.planes[3].grid_stacks[1].sub[0] == pytest.approx(np.array([coord1, coord2]), 1e-4)
            assert tracer.planes[3].grid_stacks[1].sub[1] == pytest.approx(np.array([coord3, 0.0]), 1e-4)
            assert tracer.planes[3].grid_stacks[1].blurring[0] == pytest.approx(np.array([coord3, 0.0]), 1e-4)

    class TestImagePlaneImages:

        def test__x2_grids__multiple_galaxies_and_planes_in__x2_images_out(self, grid_stack_0, grid_stack_1):
            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))
            g3 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.4))
            g4 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.5))

            tracer = ray_tracing_stack.TracerMultiPlanesStack(galaxies=[g0, g1, g2, g3, g4],
                                                        image_plane_grid_stacks=[grid_stack_0, grid_stack_1],
                                                        cosmology=cosmo.Planck15)

            plane_0 = plane_stack.PlaneStack(galaxies=[g0, g3], grid_stacks=[grid_stack_0, grid_stack_1],
                                    compute_deflections=True)
            plane_1 = plane_stack.PlaneStack(galaxies=[g1, g4], grid_stacks=[grid_stack_0, grid_stack_1],
                                    compute_deflections=True)
            plane_2 = plane_stack.PlaneStack(galaxies=[g2], grid_stacks=[grid_stack_0, grid_stack_1],
                                    compute_deflections=False)

            image_plane_image_0 = plane_0.image_plane_images[0] + plane_1.image_plane_images[0] + \
                                  plane_2.image_plane_images[0]

            assert image_plane_image_0.shape == (3, 4)
            assert (image_plane_image_0 == tracer.image_plane_images[0]).all()

            image_plane_image_1 = plane_0.image_plane_images[1] + plane_1.image_plane_images[1] + \
                                  plane_2.image_plane_images[1]

            assert image_plane_image_1.shape == (3, 4)
            assert (image_plane_image_1 == tracer.image_plane_images[1]).all()

        def test__padded_2d_image_from_plane__mapped_correctly(self, padded_grid_stack):
            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))

            tracer = ray_tracing_stack.TracerMultiPlanesStack(galaxies=[g0, g1, g2],
                                                        image_plane_grid_stacks=[padded_grid_stack, padded_grid_stack],
                                                        cosmology=cosmo.Planck15)

            plane_0 = plane_stack.PlaneStack(galaxies=[g0], grid_stacks=[padded_grid_stack, padded_grid_stack],
                                    compute_deflections=True)
            plane_1 = plane_stack.PlaneStack(galaxies=[g1], grid_stacks=[padded_grid_stack, padded_grid_stack],
                                    compute_deflections=True)
            plane_2 = plane_stack.PlaneStack(galaxies=[g2], grid_stacks=[padded_grid_stack, padded_grid_stack],
                                    compute_deflections=False)

            image_plane_image = plane_0.image_plane_images[0] + plane_1.image_plane_images[0] + \
                                plane_2.image_plane_images[0]

            assert image_plane_image.shape == (1, 2)
            assert (image_plane_image == tracer.image_plane_images[0]).all()

            image_plane_image = plane_0.image_plane_images[1] + plane_1.image_plane_images[1] + \
                                plane_2.image_plane_images[1]

            assert image_plane_image.shape == (1, 2)
            assert (image_plane_image == tracer.image_plane_images[1]).all()

        def test__padded_2d_image_for_simulation__mapped_correctly_not_trimmed(self, padded_grid_stack):
            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))

            tracer = ray_tracing_stack.TracerMultiPlanesStack(galaxies=[g0, g1, g2],
                                                        image_plane_grid_stacks=[padded_grid_stack, padded_grid_stack],
                                                        cosmology=cosmo.Planck15)

            plane_0 = plane_stack.PlaneStack(galaxies=[g0], grid_stacks=[padded_grid_stack, padded_grid_stack],
                                    compute_deflections=True)
            plane_1 = plane_stack.PlaneStack(galaxies=[g1], grid_stacks=[padded_grid_stack, padded_grid_stack],
                                    compute_deflections=True)
            plane_2 = plane_stack.PlaneStack(galaxies=[g2], grid_stacks=[padded_grid_stack, padded_grid_stack],
                                    compute_deflections=False)

            image_plane_image_for_simulation = plane_0.image_plane_images_for_simulation[0] + \
                                               plane_1.image_plane_images_for_simulation[0] + \
                                               plane_2.image_plane_images_for_simulation[0]

            assert image_plane_image_for_simulation.shape == (3, 4)
            assert (image_plane_image_for_simulation == tracer.image_plane_images_for_simulation[0]).all()

            image_plane_image_for_simulation = plane_0.image_plane_images_for_simulation[1] + \
                                               plane_1.image_plane_images_for_simulation[1] + \
                                               plane_2.image_plane_images_for_simulation[1]

            assert image_plane_image_for_simulation.shape == (3, 4)
            assert (image_plane_image_for_simulation == tracer.image_plane_images_for_simulation[1]).all()

    class TestImagePlaneBlurringImages:

        def test__many_galaxies__x2_grids_in__x2_images_out(self, grid_stack_0, grid_stack_1):
            sersic = lp.EllipticalSersic(axis_ratio=0.5, phi=0.0, intensity=1.0, effective_radius=0.6, sersic_index=4.0)

            g0 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g1 = g.Galaxy(redshift=1.0, light_profile=sersic)
            g2 = g.Galaxy(redshift=2.0, light_profile=sersic)
            g3 = g.Galaxy(redshift=0.1, light_profile=sersic)
            g4 = g.Galaxy(redshift=1.0, light_profile=sersic)

            tracer = ray_tracing_stack.TracerMultiPlanesStack(galaxies=[g0, g1, g2, g3, g4],
                                                        image_plane_grid_stacks=[grid_stack_0, grid_stack_1],
                                                        cosmology=cosmo.Planck15)

            plane_0 = plane_stack.PlaneStack(galaxies=[g0, g3], grid_stacks=[grid_stack_0, grid_stack_1],
                                    compute_deflections=True)
            plane_1 = plane_stack.PlaneStack(galaxies=[g1, g4], grid_stacks=[grid_stack_0, grid_stack_1],
                                    compute_deflections=True)
            plane_2 = plane_stack.PlaneStack(galaxies=[g2], grid_stacks=[grid_stack_0, grid_stack_1],
                                    compute_deflections=False)

            image_plane_blurring_image_0 = plane_0.image_plane_blurring_images_1d[0] + \
                                           plane_1.image_plane_blurring_images_1d[0] + \
                                           plane_2.image_plane_blurring_images_1d[0]

            assert (image_plane_blurring_image_0 == tracer.image_plane_blurring_images_1d[0]).all()

            image_plane_blurring_image_1 = plane_0.image_plane_blurring_images_1d[1] + \
                                           plane_1.image_plane_blurring_images_1d[1] + \
                                           plane_2.image_plane_blurring_images_1d[1]

            assert (image_plane_blurring_image_1 == tracer.image_plane_blurring_images_1d[1]).all()

        def test__galaxies_have_no_pixelization__no_pix_grid_added(self):
            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]]), pixel_scale=1.0)

            grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                            psf_shape=(1, 1))

            tracer = ray_tracing_stack.TracerMultiPlanesStack(galaxies=[g.Galaxy(redshift=2.0), g.Galaxy(redshift=1.0)],
                                                        image_plane_grid_stacks=[grid_stack_0, grid_stack_0])

            assert (tracer.planes[0].grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()

            assert (tracer.planes[0].grid_stacks[1].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grid_stacks[1].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxies_have_other_pixelization__returns_normal_grids(self):
            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, False, False]]), pixel_scale=1.0)

            grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                            psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.Rectangular(shape=(3, 3)),
                              regularization=regularization.Constant(), redshift=2.0)

            tracer = ray_tracing_stack.TracerMultiPlanesStack(galaxies=[galaxy, g.Galaxy(redshift=1.0)],
                                                        image_plane_grid_stacks=[grid_stack_0, grid_stack_0])

            assert (tracer.planes[0].grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()

            assert (tracer.planes[0].grid_stacks[1].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grid_stacks[1].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__but_grid_is_padded_grid__returns_normal_grids(self):
            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, True, False]]), pixel_scale=1.0)

            grid_stack_0 = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma,
                                                                                                   sub_grid_size=1,
                                                                                                   psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant(), redshift=2.0)

            tracer = ray_tracing_stack.TracerMultiPlanesStack(galaxies=[galaxy, g.Galaxy(redshift=1.0)],
                                                        image_plane_grid_stacks=[grid_stack_0, grid_stack_0])

            assert (tracer.planes[0].grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grid_stacks[0].pix == np.array([[0.0, 0.0]])).all()

            assert (tracer.planes[0].grid_stacks[1].pix == np.array([[0.0, 0.0]])).all()
            assert (tracer.planes[1].grid_stacks[1].pix == np.array([[0.0, 0.0]])).all()

        def test__setup_pixelization__galaxy_has_pixelization__returns_grids_with_pix_grid(self):
            ma = mask.Mask(np.array([[False, False, False],
                                     [False, False, False],
                                     [False, True, False]]), pixel_scale=1.0)

            grid_stack_0 = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=ma, sub_grid_size=1,
                                                                                            psf_shape=(1, 1))

            galaxy = g.Galaxy(pixelization=pixelizations.AdaptiveMagnification(shape=(3, 3)),
                              regularization=regularization.Constant(), redshift=2.0)

            tracer = ray_tracing_stack.TracerMultiPlanesStack(galaxies=[galaxy, g.Galaxy(redshift=1.0)],
                                                        image_plane_grid_stacks=[grid_stack_0, grid_stack_0])

            assert (tracer.planes[0].grid_stacks[0].regular == grid_stack_0.regular).all()
            assert (tracer.planes[0].grid_stacks[0].sub == grid_stack_0.sub).all()
            assert (tracer.planes[0].grid_stacks[0].blurring == grid_stack_0.blurring).all()
            assert (tracer.planes[0].grid_stacks[0].pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                                     [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                     [-1.0, -1.0], [-1.0, 1.0]])).all()

            assert (tracer.planes[1].grid_stacks[0].regular == grid_stack_0.regular).all()
            assert (tracer.planes[1].grid_stacks[0].sub == grid_stack_0.sub).all()
            assert (tracer.planes[1].grid_stacks[0].blurring == grid_stack_0.blurring).all()
            assert (tracer.planes[1].grid_stacks[0].pix == np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
                                                                     [0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
                                                                     [-1.0, -1.0], [-1.0, 1.0]])).all()

    class TestCompareToNonStacks:

        def test__compare_all_quantities_to_non_stack_tracers(self, grid_stack_0, grid_stack_1):
            g0 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.1))
            g1 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.2))
            g2 = g.Galaxy(redshift=2.0, light_profile=lp.EllipticalSersic(intensity=0.3))
            g3 = g.Galaxy(redshift=0.1, light_profile=lp.EllipticalSersic(intensity=0.4))
            g4 = g.Galaxy(redshift=1.0, light_profile=lp.EllipticalSersic(intensity=0.5))

            tracer_stack = ray_tracing_stack.TracerMultiPlanesStack(galaxies=[g0, g1, g2, g3, g4],
                                                              image_plane_grid_stacks=[grid_stack_0, grid_stack_1])

            tracer_0 = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2, g3, g4],
                                                     image_plane_grid_stack=grid_stack_0)

            assert (tracer_stack.image_plane_images[0] == tracer_0.image_plane_image).all()
            assert (tracer_stack.image_plane_blurring_images_1d[0] == tracer_0.image_plane_blurring_image_1d).all()

            tracer_1 = ray_tracing.TracerMultiPlanes(galaxies=[g0, g1, g2, g3, g4],
                                                     image_plane_grid_stack=grid_stack_1)

            assert (tracer_stack.image_plane_images[1] == tracer_1.image_plane_image).all()
            assert (tracer_stack.image_plane_blurring_images_1d[1] == tracer_1.image_plane_blurring_image_1d).all()