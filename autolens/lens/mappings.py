"""
How a region of the source plane maps to the image plane, for any kind of source.

A **mapping** pairs one source-plane region with the image-plane regions (the multiple
images) it maps to. `autoarray.inversion.mappings` builds one for a *pixelized* source
from a `Mapper`'s mapping matrix (engine A); this module builds one for a *parametric*
source from `ShapeSolver`'s ray-traced triangles (engine B), and dispatches between the
two given a `FitImaging`, so a caller does not have to know which kind of source a model
used.

Both engines return the same `Mapping` and `ImageRegion` objects, so everything
downstream -- `subplot_mappings`, the brightest-pixel positions spectroscopic follow-up
needs, the per-image magnifications -- is written once.

Everything here is pure numpy and is never jitted: it is diagnostic and visualization
code which runs once per figure or once per result, not inside a likelihood.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

import autoarray as aa
import autogalaxy as ag

from autoarray.structures.triangles.shape import Circle, Shape

from autolens.lens.tracer import Tracer
from autolens.point.solver.shape_solver import ShapeSolver

logger = logging.getLogger(__name__)


def source_mapping_from(
    tracer: Tracer,
    grid: aa.type.Grid2DLike,
    shape: Shape,
    plane_index: int = -1,
    plane_redshift: Optional[float] = None,
    pixel_scale_precision: Optional[float] = None,
    magnification_threshold: float = 0.1,
) -> aa.Mapping:
    """
    The `Mapping` of a source-plane `Shape` through a tracer, via `ShapeSolver` (engine B).

    This is the thin entry point: it builds the solver for `grid` and delegates to
    `ShapeSolver.mapping_from`.

    Parameters
    ----------
    tracer
        The tracer which ray-traces the image plane to the source plane.
    grid
        The image-plane grid whose extent is tiled with triangles and whose pixels the
        image regions are reported on.
    shape
        The source-plane region, e.g. a `Circle` at a light profile's centre.
    plane_index
        The index of the plane `shape` lives in, used only when `plane_redshift` is `None`.
    plane_redshift
        The redshift of the plane `shape` lives in. `None` derives it from `plane_index`.
    pixel_scale_precision
        The triangle size the solver refines down to. `None` uses a tenth of the grid's
        pixel scale, which resolves the image of a source a few pixels across to well
        inside a pixel while keeping the solve fast.
    magnification_threshold
        Images whose magnification does not exceed this are discarded, which removes the
        formally-infinitely-demagnified central image of a singular profile.

    Returns
    -------
    The mapping of the shape.
    """
    if pixel_scale_precision is None:
        pixel_scale_precision = float(grid.pixel_scale) / 10.0

    if plane_redshift is None:
        plane_redshift = float(tracer.plane_redshifts[plane_index])

    solver = ShapeSolver.for_grid(
        grid=grid,
        pixel_scale_precision=pixel_scale_precision,
        magnification_threshold=magnification_threshold,
    )

    return solver.mapping_from(
        tracer=tracer,
        shape=shape,
        plane_redshift=plane_redshift,
    )


def traced_region_from(
    tracer: Tracer,
    region_grid: aa.type.Grid2DLike,
    plane_index: int = -1,
) -> np.ndarray:
    """
    The source-plane outline of an image-plane region, as the convex hull of its traced
    coordinates.

    An image-plane region is a blob of pixels; ray-tracing it gives a cloud of source-plane
    coordinates whose outline is what the source-plane panel of a mapping figure draws. A
    convex hull is used rather than an exact outline because the traced cloud of one
    multiple image is a small, roughly convex patch of the source plane -- it is the *image*
    which is stretched into an arc, not its source.

    Parameters
    ----------
    tracer
        The tracer which ray-traces the image plane to the source plane.
    region_grid
        The ``(N, 2)`` ``(y, x)`` image-plane coordinates of the region, e.g. an
        `ImageRegion`'s `scaled_coordinates`.
    plane_index
        The plane the coordinates are traced to.

    Returns
    -------
    An ``(M, 2)`` array of ``(y, x)`` source-plane coordinates forming a closed loop
    (its first row is repeated as its last). Fewer than three traced coordinates, or a
    degenerate (collinear) cloud, returns the traced coordinates themselves.
    """
    from scipy.spatial import ConvexHull, QhullError

    if not isinstance(region_grid, aa.Grid2DIrregular):
        region_grid = aa.Grid2DIrregular(values=np.asarray(region_grid))

    traced = np.asarray(
        tracer.traced_grid_2d_list_from(grid=region_grid)[plane_index].array
    )

    if traced.shape[0] < 3:
        return traced

    try:
        hull = ConvexHull(traced)
    except QhullError:
        return traced

    vertices = traced[hull.vertices]

    return np.vstack([vertices, vertices[:1]])


def _light_profile_shape_from(tracer: Tracer, plane_index: int) -> Circle:
    """
    The default source-plane `Circle` for a parametric source: centred on the first light
    profile of the plane, with a radius of its half-light radius.

    Parameters
    ----------
    tracer
        The tracer whose plane holds the source.
    plane_index
        The index of the source plane.

    Returns
    -------
    The circle.
    """
    light_profiles = tracer.planes[plane_index].cls_list_from(cls=ag.LightProfile)

    if len(light_profiles) == 0:
        raise ValueError(
            f"Plane {plane_index} of the tracer has neither a pixelization nor a light "
            f"profile, so there is no source region to map. Pass `shape=` explicitly (e.g. "
            f"`shape=aa.Circle(y, x, radius=...)`) to say which source-plane region you "
            f"want the images of."
        )

    profile = light_profiles[0]

    radius = profile.half_light_radius

    if radius is None:
        raise ValueError(
            f"The source light profile {type(profile).__name__} has no `half_light_radius` "
            f"(it defines no `effective_radius`), so the size of the source region cannot be "
            f"inferred from it. Pass `shape=` explicitly, e.g. "
            f"`shape=aa.Circle({profile.centre[0]}, {profile.centre[1]}, radius=...)`."
        )

    # `Circle` reads element 0 of a coordinate pair first, and element 0 of every PyAuto
    # grid is `y`, so a profile centred at `(y_c, x_c)` gives `Circle(y_c, x_c, radius)`
    # positionally. See the `Shape` class docstring in PyAutoArray.
    return Circle(
        float(profile.centre[0]), float(profile.centre[1]), radius=float(radius)
    )


def mappings_from_fit(
    fit,
    plane_index: int = -1,
    shape: Optional[Shape] = None,
    grid: Optional[aa.type.Grid2DLike] = None,
    pixel_scale_precision: Optional[float] = None,
    magnification_threshold: float = 0.1,
    **clump_kwargs,
) -> List[aa.Mapping]:
    """
    The mappings of a fit's source plane, whichever kind of source the model used.

    A plane which `has(cls=aa.Pixelization)` is mapped by **engine A**: the inversion's
    reconstruction is split into clumps and each clump's image-plane regions come from the
    mapper's mapping matrix. Any other plane is mapped by **engine B**: a `Circle` at the
    first light profile's centre, with its half-light radius, is traced by `ShapeSolver`.

    Parameters
    ----------
    fit
        A `FitImaging` (or any fit exposing `tracer`, `mask` and, for engine A, `inversion`).
    plane_index
        The index of the source plane to map.
    shape
        The source-plane region for engine B. `None` builds it from the plane's first light
        profile. Ignored by engine A, which finds its own clumps.
    grid
        The image-plane grid engine B tiles. `None` uses the fit's unmasked grid.
    pixel_scale_precision
        The triangle size engine B refines to; `None` uses a tenth of the pixel scale.
    magnification_threshold
        Engine B discards images below this magnification.
    clump_kwargs
        `threshold`, `min_pixels`, `total_clumps`, `pix_indexes` and `weight_threshold`,
        passed to engine A's `Inversion.mappings_from`. Ignored by engine B.

    Returns
    -------
    One `Mapping` per source-plane region: one per clump for engine A, one for engine B.
    """
    tracer = fit.tracer_linear_light_profiles_to_light_profiles

    if tracer.planes[plane_index].has(cls=aa.Pixelization):
        weight_threshold = clump_kwargs.pop("weight_threshold", 0.0)

        mapper_index = _mapper_index_from(fit=fit, plane_index=plane_index)

        return fit.inversion.mappings_from(
            mapper_index=mapper_index,
            weight_threshold=weight_threshold,
            **clump_kwargs,
        )

    if shape is None:
        shape = _light_profile_shape_from(tracer=tracer, plane_index=plane_index)

    if grid is None:
        grid = fit.mask.derive_grid.all_false

    return [
        source_mapping_from(
            tracer=tracer,
            grid=grid,
            shape=shape,
            plane_index=plane_index,
            pixel_scale_precision=pixel_scale_precision,
            magnification_threshold=magnification_threshold,
        )
    ]


def _mapper_index_from(fit, plane_index: int) -> int:
    """
    The index into the inversion's mappers of the mapper belonging to `plane_index`.

    A fit has one mapper per pixelized plane and planes are ordered by redshift, so the
    mapper of plane ``i`` is mapper ``i - 1`` for the usual lens-plus-source fit. The
    normalisation of a negative `plane_index` is done here so the two agree.

    Parameters
    ----------
    fit
        The fit whose inversion holds the mappers.
    plane_index
        The index of the source plane.

    Returns
    -------
    The mapper index, which is 0 whenever there is only one mapper.
    """
    mapper_list = fit.inversion.cls_list_from(cls=aa.Mapper)

    if len(mapper_list) == 1:
        return 0

    total_planes = len(fit.tracer.planes)

    plane_index = plane_index % total_planes

    return max(plane_index - 1, 0)


def multiple_image_positions_from(
    fit,
    plane_index: int = -1,
    use_centroid: bool = False,
    **mapping_kwargs,
) -> aa.Grid2DIrregular:
    """
    The brightest image-plane coordinate of each multiple image of the source.

    This is the quantity spectroscopic follow-up needs: where to point a fibre on each
    lensed image. The brightness is read from the *model* image of the source plane, so
    the positions describe the lens model rather than the noise realisation of the data.

    Parameters
    ----------
    fit
        The `FitImaging` whose source is mapped.
    plane_index
        The index of the source plane.
    use_centroid
        If `True` each position is the flux-weighted centroid of its image region rather
        than its brightest pixel. The centroid is sub-pixel and smooth under small model
        changes; the brightest pixel is exactly a data pixel, which is what a fibre is
        pointed at.
    mapping_kwargs
        Passed to `mappings_from_fit`.

    Returns
    -------
    A `Grid2DIrregular` of ``(y, x)`` arcsec coordinates, one per multiple image, ordered
    as the image regions are (largest image first, mapping by mapping).
    """
    model_image = fit.model_images_of_planes_list[plane_index]

    mappings = mappings_from_fit(fit=fit, plane_index=plane_index, **mapping_kwargs)

    positions = []

    for mapping in mappings:
        for region in mapping.image_regions:
            if len(region.slim_indexes) == 0:
                continue

            if use_centroid:
                positions.append(region.centroid_from(array=model_image))
            else:
                positions.append(region.brightest_coordinate_from(array=model_image))

    if len(positions) == 0:
        # A shape-(0,) irregular grid would not be indexable as [:, 0], so an empty
        # result keeps the (N, 2) contract of a non-empty one.
        return aa.Grid2DIrregular(values=np.zeros(shape=(0, 2)))

    return aa.Grid2DIrregular(values=positions)


def multiple_image_pixel_coordinates_from(
    fit,
    plane_index: int = -1,
    use_centroid: bool = False,
    **mapping_kwargs,
) -> List[Tuple[float, float]]:
    """
    The multiple-image positions as pixel coordinates of the fit's data.

    The library deliberately stops at pixel coordinates. Turning them into RA / Dec is a
    WCS question, which depends on the header of the FITS the data came from rather than
    on anything the lens model knows, and is shown inline in the mappings guide with
    astropy.

    Parameters
    ----------
    fit
        The `FitImaging` whose source is mapped.
    plane_index
        The index of the source plane.
    use_centroid
        As `multiple_image_positions_from`.
    mapping_kwargs
        Passed to `mappings_from_fit`.

    Returns
    -------
    One ``(y, x)`` pixel coordinate per multiple image, in the WCS/FITS convention: 1-based,
    referred to pixel centres, and continuous (the fractional part is the sub-pixel offset).
    """
    positions = multiple_image_positions_from(
        fit=fit,
        plane_index=plane_index,
        use_centroid=use_centroid,
        **mapping_kwargs,
    )

    geometry = fit.dataset.data.mask.geometry

    return [
        geometry.pixel_coordinates_wcs_2d_from(scaled_coordinates_2d=tuple(position))
        for position in np.asarray(positions.array)
    ]


def magnifications_from(
    fit,
    plane_index: int = -1,
    shape: Optional[Shape] = None,
    grid: Optional[aa.type.Grid2DLike] = None,
    pixel_scale_precision: Optional[float] = None,
    magnification_threshold: float = 0.1,
    **clump_kwargs,
) -> List[float]:
    """
    The magnification of each multiple image of the source.

    The two engines measure it differently, because they know different things:

    - **Engine B** (parametric) measures it geometrically, as
      `ShapeSolver.find_magnification(per_image=True)`: the image-plane area the source
      shape's images cover, divided by the shape's own area. This is the true
      magnification of the lens model, independent of the source's surface brightness.
    - **Engine A** (pixelized) has no source-plane *area* to divide by, only a mesh, so it
      measures the flux ratio instead: each image region's summed model flux divided by
      the total model flux of the whole clump across all of its images. That is the
      *relative* magnification of the images of one clump -- it sums to 1 over a clump's
      images -- and is not on the same scale as engine B's absolute magnification.

    Parameters
    ----------
    fit
        The `FitImaging` whose source is mapped.
    plane_index
        The index of the source plane.
    shape, grid, pixel_scale_precision, magnification_threshold
        As `mappings_from_fit` (engine B only).
    clump_kwargs
        As `mappings_from_fit` (engine A only).

    Returns
    -------
    One magnification per multiple image, ordered as `multiple_image_positions_from` is.
    """
    tracer = fit.tracer_linear_light_profiles_to_light_profiles

    if not tracer.planes[plane_index].has(cls=aa.Pixelization):
        if shape is None:
            shape = _light_profile_shape_from(tracer=tracer, plane_index=plane_index)

        if grid is None:
            grid = fit.mask.derive_grid.all_false

        if pixel_scale_precision is None:
            pixel_scale_precision = float(grid.pixel_scale) / 10.0

        solver = ShapeSolver.for_grid(
            grid=grid,
            pixel_scale_precision=pixel_scale_precision,
            magnification_threshold=magnification_threshold,
        )

        return solver.find_magnification(
            tracer=tracer,
            shape=shape,
            plane_redshift=float(tracer.plane_redshifts[plane_index]),
            per_image=True,
        )

    model_image = fit.model_images_of_planes_list[plane_index]

    magnifications = []

    for mapping in mappings_from_fit(fit=fit, plane_index=plane_index, **clump_kwargs):
        fluxes = [
            region.flux_from(array=model_image) for region in mapping.image_regions
        ]

        total = float(np.sum(fluxes))

        magnifications += [
            float(flux / total) if total != 0.0 else 0.0 for flux in fluxes
        ]

    return magnifications
