"""
Tests for `autolens.lens.mappings` -- the two engines which pair a source-plane region
with the image-plane regions (the multiple images) it maps to, and the fit-level
quantities built on them.
"""

import numpy as np
import pytest

import autoarray as aa
import autolens as al

from autoarray.structures.triangles.shape import Circle

from autolens.lens import mappings


EINSTEIN_RADIUS = 1.0


@pytest.fixture
def grid():
    return al.Grid2D.uniform(shape_native=(80, 80), pixel_scales=0.05)


@pytest.fixture
def tracer():
    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(
                    centre=(0.0, 0.0), einstein_radius=EINSTEIN_RADIUS
                ),
            ),
            al.Galaxy(
                redshift=1.0,
                light=al.lp.SersicSph(
                    centre=(0.2, 0.0), intensity=1.0, effective_radius=0.1
                ),
            ),
        ]
    )


def test__source_inside_the_caustic__maps_to_two_images(tracer, grid):
    """
    An isothermal sphere's caustic is the point ``beta = 0``, so any source inside its
    Einstein radius is multiply imaged: two images, one on each side of the lens.
    """
    mapping = mappings.source_mapping_from(
        tracer=tracer,
        grid=grid,
        shape=Circle(0.2, 0.0, radius=0.05),
        pixel_scale_precision=0.005,
    )

    assert len(mapping.image_regions) == 2

    y_coordinates = sorted(region.centre[0] for region in mapping.image_regions)

    assert y_coordinates[0] == pytest.approx(0.2 - EINSTEIN_RADIUS, abs=0.05)
    assert y_coordinates[1] == pytest.approx(0.2 + EINSTEIN_RADIUS, abs=0.05)


def test__source_outside_the_einstein_radius__maps_to_one_image(tracer):
    """
    A source further from the lens than its Einstein radius has a single image, and it
    sits beyond the Einstein radius on the same side as the source.

    The grid is wider than the fixture's: the single image of a source at ``beta = 1.5``
    lies at ``beta + theta_E = 2.5``, which a +/- 2" grid does not contain -- and an image
    outside the tiled extent is not found, correctly, since the solver only searches the
    image plane it was given.
    """
    mapping = mappings.source_mapping_from(
        tracer=tracer,
        grid=al.Grid2D.uniform(shape_native=(120, 120), pixel_scales=0.05),
        shape=Circle(1.5, 0.0, radius=0.05),
        pixel_scale_precision=0.005,
    )

    assert len(mapping.image_regions) == 1
    assert mapping.image_regions[0].centre[0] == pytest.approx(
        1.5 + EINSTEIN_RADIUS, abs=0.05
    )


def test__traced_region_is_a_closed_source_plane_loop(tracer, grid):
    """
    An image-plane region traced back to the source plane, outlined by its convex hull.
    """
    mapping = mappings.source_mapping_from(
        tracer=tracer,
        grid=grid,
        shape=Circle(0.2, 0.0, radius=0.05),
        pixel_scale_precision=0.005,
    )

    region = mapping.image_regions[0]

    hull = mappings.traced_region_from(
        tracer=tracer, region_grid=region.scaled_coordinates
    )

    assert hull.ndim == 2
    assert hull.shape[1] == 2
    assert hull[0] == pytest.approx(hull[-1])

    # An image traces back onto its source, so the hull must sit around the shape.
    assert np.mean(hull[:, 0]) == pytest.approx(0.2, abs=0.1)


def test__traced_region_of_fewer_than_three_points_is_returned_unchanged(tracer):
    """
    A convex hull needs three points; two traced coordinates are returned as they are
    rather than raising out of a figure.
    """
    traced = mappings.traced_region_from(
        tracer=tracer, region_grid=np.array([[1.0, 0.0], [-1.0, 0.0]])
    )

    assert traced.shape == (2, 2)


# ------------------------------------------------------------------------------------ #
# engine dispatch
# ------------------------------------------------------------------------------------ #


def test__parametric_fit__dispatches_to_the_shape_solver(fit_imaging_x2_plane_7x7):
    """
    A plane with no pixelization is mapped by engine B, which returns exactly one
    `Mapping` (the one source shape) with no mesh pixel indexes and the shape's own
    boundary as its source contour.
    """
    results = mappings.mappings_from_fit(fit=fit_imaging_x2_plane_7x7)

    assert len(results) == 1

    mapping = results[0]

    assert isinstance(mapping, aa.Mapping)
    assert len(mapping.pix_indexes) == 0
    assert mapping.peak_value is None
    assert len(mapping.source_contours) == 1


def test__pixelized_fit__dispatches_to_the_inversion(
    fit_imaging_x2_plane_inversion_7x7,
):
    """
    A plane which `has(cls=aa.Pixelization)` is mapped by engine A, whose mappings carry
    mesh pixel indexes and a peak reconstructed value -- the two fields engine B leaves
    empty. The 7x7 fixture reconstructs to all zeros, so clump finding legitimately finds
    nothing at any threshold; `pix_indexes` is the documented bypass (the tutorial path)
    and is what exercises the dispatch here.
    """
    results = mappings.mappings_from_fit(
        fit=fit_imaging_x2_plane_inversion_7x7, pix_indexes=[[0, 1, 2]]
    )

    assert len(results) == 1

    mapping = results[0]

    assert isinstance(mapping, aa.Mapping)
    assert list(mapping.pix_indexes) == [0, 1, 2]
    assert mapping.peak_value is not None


def test__source_plane_with_no_light_profile__names_the_shape_input(
    masked_imaging_7x7,
):
    """
    Neither engine can guess a region for a source plane which has neither a pixelization
    nor a light profile, so the error must point at `shape=`.
    """
    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=al.Tracer(
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
                ),
                al.Galaxy(redshift=1.0),
            ]
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        mappings.mappings_from_fit(fit=fit)

    assert "shape=" in str(exc_info.value)


# ------------------------------------------------------------------------------------ #
# fit-level quantities
# ------------------------------------------------------------------------------------ #


def test__multiple_image_positions__one_per_image_region(fit_imaging_x2_plane_7x7):
    positions = mappings.multiple_image_positions_from(fit=fit_imaging_x2_plane_7x7)

    total_regions = sum(
        len(mapping.image_regions)
        for mapping in mappings.mappings_from_fit(fit=fit_imaging_x2_plane_7x7)
    )

    assert isinstance(positions, aa.Grid2DIrregular)
    assert np.asarray(positions.array).shape == (total_regions, 2)


def test__multiple_image_positions__centroid_and_brightest_pixel_agree_roughly(
    tracer, grid
):
    """
    On a real (rather than 7x7 fixture) lens both routes must land on the same image, the
    centroid sub-pixel and the brightest pixel snapped to the data grid.
    """
    dataset = _simulated_dataset(tracer=tracer, grid=grid)

    fit = al.FitImaging(dataset=dataset, tracer=tracer)

    brightest = np.asarray(
        mappings.multiple_image_positions_from(
            fit=fit, pixel_scale_precision=0.01
        ).array
    )
    centroids = np.asarray(
        mappings.multiple_image_positions_from(
            fit=fit, use_centroid=True, pixel_scale_precision=0.01
        ).array
    )

    assert brightest.shape == centroids.shape == (2, 2)
    assert brightest == pytest.approx(centroids, abs=0.1)


def test__multiple_image_pixel_coordinates__are_inside_the_data(tracer, grid):
    dataset = _simulated_dataset(tracer=tracer, grid=grid)

    fit = al.FitImaging(dataset=dataset, tracer=tracer)

    pixel_coordinates = mappings.multiple_image_pixel_coordinates_from(
        fit=fit, pixel_scale_precision=0.01
    )

    assert len(pixel_coordinates) == 2

    shape_native = fit.dataset.data.mask.shape_native

    for y, x in pixel_coordinates:
        # WCS pixel coordinates are 1-based and refer to pixel centres.
        assert 1.0 <= y <= shape_native[0]
        assert 1.0 <= x <= shape_native[1]


def test__magnifications__parametric_matches_the_analytic_isothermal_sphere(
    tracer, grid
):
    """
    Engine B's magnifications are absolute, so they can be checked against the analytic
    isothermal sphere: ``1 + theta_E / beta`` and ``theta_E / beta - 1`` for beta = 0.2.
    """
    dataset = _simulated_dataset(tracer=tracer, grid=grid)

    fit = al.FitImaging(dataset=dataset, tracer=tracer)

    magnifications = mappings.magnifications_from(
        fit=fit, shape=Circle(0.2, 0.0, radius=0.05), pixel_scale_precision=0.005
    )

    assert len(magnifications) == 2

    assert magnifications[0] == pytest.approx(1.0 + EINSTEIN_RADIUS / 0.2, rel=0.02)
    assert magnifications[1] == pytest.approx(EINSTEIN_RADIUS / 0.2 - 1.0, rel=0.02)


def test__magnifications__pixelized_are_flux_shares_summing_to_one(grid):
    """
    Engine A has no source-plane area to divide by, so it reports each image's share of
    the clump's total model flux -- which sums to one per clump, by construction, and is
    documented as being on a different scale from engine B's absolute magnification.

    This needs a reconstruction with something in it, so it fits a real (if small)
    simulated lens with a rectangular pixelization rather than using the 7x7 fixture,
    whose reconstruction is all zeros.
    """
    fit = _pixelized_fit(grid=grid)

    clump_kwargs = dict(threshold=0.5, min_pixels=1)

    magnifications = mappings.magnifications_from(fit=fit, **clump_kwargs)

    total_clumps = len(mappings.mappings_from_fit(fit=fit, **clump_kwargs))

    assert total_clumps > 0
    assert len(magnifications) >= total_clumps
    assert sum(magnifications) == pytest.approx(total_clumps, abs=1.0e-6)


def _pixelized_fit(grid):
    """
    A `FitImaging` of a simulated lens whose source is a rectangular pixelization, so that
    engine A has a non-zero reconstruction to find clumps in.
    """
    source_tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(
                    centre=(0.0, 0.0), einstein_radius=EINSTEIN_RADIUS
                ),
            ),
            al.Galaxy(
                redshift=1.0,
                light=al.lp.SersicSph(
                    centre=(0.1, 0.0), intensity=1.0, effective_radius=0.2
                ),
            ),
        ]
    )

    dataset = _simulated_dataset(tracer=source_tracer, grid=grid)

    pixelized_tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalSph(
                    centre=(0.0, 0.0), einstein_radius=EINSTEIN_RADIUS
                ),
            ),
            al.Galaxy(
                redshift=1.0,
                pixelization=al.Pixelization(
                    mesh=al.mesh.RectangularUniform(shape=(10, 10)),
                    regularization=al.reg.Constant(coefficient=1.0),
                ),
            ),
        ]
    )

    return al.FitImaging(dataset=dataset, tracer=pixelized_tracer)


def _simulated_dataset(tracer, grid):
    """
    A noiseless simulated dataset of `tracer`, masked to the grid's extent.

    Noiseless so that the brightest pixel of a model image is the brightest pixel of the
    data, which is what makes the position assertions above deterministic.
    """
    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=al.Convolver.from_gaussian(
            shape_native=(3, 3),
            pixel_scales=grid.pixel_scales[0],
            sigma=0.05,
            normalize=True,
        ),
        add_poisson_noise_to_data=False,
    )

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    dataset.noise_map = al.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=grid.pixel_scales
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.data.shape_native,
        pixel_scales=grid.pixel_scales,
        radius=1.9,
    )

    return dataset.apply_mask(mask=mask)
