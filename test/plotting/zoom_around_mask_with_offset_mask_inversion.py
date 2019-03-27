from autolens.data.array import mask as msk
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.galaxy import galaxy as g
from autolens.lens import ray_tracing
from autolens.lens import lens_fit
from autolens.lens import lens_data as ld
from autolens.data.plotters import ccd_plotters
from autolens.lens.plotters import lens_fit_plotters
from test.simulation import simulation_util

ccd_data = simulation_util.load_test_ccd_data(data_type='no_lens_light_and_source_smooth_offset_centre',
                                              data_resolution='LSST')

def mask_and_fit_with_offset_centre(centre):

    mask = msk.Mask.elliptical(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, major_axis_radius_arcsec=3.0,
                               axis_ratio=1.0, phi=0.0, centre=centre)

    # The lines of code below do everything we're used to, that is, setup an image and its grid stack, mask it, trace it
    # via a tracer, setup the rectangular mapper, etc.
    lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(1.0, 1.0), einstein_radius=1.6,
                                                            axis_ratio=0.7, phi=45.0))
    source_galaxy = g.Galaxy(pixelization=pix.AdaptiveMagnification(shape=(20, 20)),
                             regularization=reg.Constant(coefficients=(1.0,)))

    lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=lens_data.grid_stack)
    fit = lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

    return mask, fit

mask, fit = mask_and_fit_with_offset_centre(centre=(1.0, 1.0))

lens_fit_plotters.plot_fit_subplot(fit=fit, should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                   positions=[[[2.2, 2.2], [-0.2, -0.2], [-0.2, 2.2], [2.2, -0.2]]],
                                   should_plot_image_plane_pix=True)

mask, fit = mask_and_fit_with_offset_centre(centre=(1.05, 1.05))


lens_fit_plotters.plot_fit_subplot(fit=fit, should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                   positions=[[[2.2, 2.2], [-0.2, -0.2], [-0.2, 2.2], [2.2, -0.2]]],
                                   should_plot_image_plane_pix=True)

mask, fit = mask_and_fit_with_offset_centre(centre=(1.1, 1.1))

lens_fit_plotters.plot_fit_subplot(fit=fit, should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                   positions=[[[2.2, 2.2], [-0.2, -0.2], [-0.2, 2.2], [2.2, -0.2]]],
                                   should_plot_image_plane_pix=True)

mask, fit = mask_and_fit_with_offset_centre(centre=(0.95, 0.95))

lens_fit_plotters.plot_fit_subplot(fit=fit, should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                   positions=[[[2.2, 2.2], [-0.2, -0.2], [-0.2, 2.2], [2.2, -0.2]]],
                                   should_plot_image_plane_pix=True)

mask, fit = mask_and_fit_with_offset_centre(centre=(0.9, 0.9))

lens_fit_plotters.plot_fit_subplot(fit=fit, should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                   positions=[[[2.2, 2.2], [-0.2, -0.2], [-0.2, 2.2], [2.2, -0.2]]],
                                   should_plot_image_plane_pix=True)