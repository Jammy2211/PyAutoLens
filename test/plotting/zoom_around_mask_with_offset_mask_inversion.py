from autolens.array import mask as msk
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.galaxy import galaxy as g
from autolens.lens import ray_tracing
from autolens.lens.lens_fit import lens_imaging_fit
from autolens.lens import lens_data as ld
from autolens.data.plotters import imaging_plotters
from autolens.lens.plotters import lens_imaging_fit_plotters
from test.simulation import simulation_util

imaging_data = simulation_util.load_test_imaging_data(
    data_type="lens_sis__source_smooth__offset_centre", data_resolution="LSST"
)


def fit_with_offset_centre(centre):

    mask = al.Mask.elliptical(
        shape=imaging_data.shape,
        pixel_scale=imaging_data.pixel_scale,
        major_axis_radius_arcsec=3.0,
        axis_ratio=1.0,
        phi=0.0,
        centre=centre,
    )

    # The lines of code below do everything we're used to, that is, setup an image and its al.ogrid, mask it, trace it
    # via a tracer, setup the rectangular mapper, etc.
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mass_profiles.EllipticalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
        ),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.pixelizations.VoronoiMagnification(shape=(20, 20)),
        regularization=al.regularization.Constant(coefficient=1.0),
    )

    lens_data = al.LensData(imaging_data=imaging_data, mask=mask)

    pixelization_grid = source_galaxy.pixelization.traced_pixelization_grids_of_planes_from_grid(
        grid=lens_data.grid
    )

    grid_stack_with_pixelization_grid = lens_data.grid.new_grid_stack_with_grids_added(
        pixelization=pixelization_grid
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy],
        image_plane_grid=grid_stack_with_pixelization_grid,
    )
    fit = al.LensImageFit.from_lens_data_and_tracer(lens_data=lens_data, tracer=tracer)

    return fit


fit = fit_with_offset_centre(centre=(1.0, 1.0))

lens_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    positions=[[[2.2, 2.2], [-0.2, -0.2], [-0.2, 2.2], [2.2, -0.2]]],
    should_plot_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(1.05, 1.05))


lens_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    positions=[[[2.2, 2.2], [-0.2, -0.2], [-0.2, 2.2], [2.2, -0.2]]],
    should_plot_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(1.1, 1.1))

lens_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    positions=[[[2.2, 2.2], [-0.2, -0.2], [-0.2, 2.2], [2.2, -0.2]]],
    should_plot_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(0.95, 0.95))

lens_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    positions=[[[2.2, 2.2], [-0.2, -0.2], [-0.2, 2.2], [2.2, -0.2]]],
    should_plot_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(5.9, 5.9))

lens_imaging_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
    positions=[[[2.2, 2.2], [-0.2, -0.2], [-0.2, 2.2], [2.2, -0.2]]],
    should_plot_image_plane_pix=True,
)
