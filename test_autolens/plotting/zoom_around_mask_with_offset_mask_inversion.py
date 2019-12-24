import autolens as al

from test_autolens.simulate.imaging import simulate_util

imaging = simulate_util.load_test_imaging(
    data_type="lens_mass__source_smooth__offset_centre", data_resolution="lsst"
)


def fit_with_offset_centre(centre):

    mask = al.mask.elliptical(
        shape_2d=imaging.shape_2d,
        pixel_scales=imaging.pixel_scales,
        major_axis_radius=3.0,
        axis_ratio=1.0,
        phi=0.0,
        centre=centre,
    )

    # The lines of code below do everything we're used to, that is, setup an image and its grid, mask it, trace it
    # via a tracer, setup the rectangular mapper, etc.
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(2.0, 2.0), einstein_radius=1.2, axis_ratio=0.7, phi=45.0
        ),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.VoronoiMagnification(shape=(20, 20)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
    fit = al.fit(masked_dataset=masked_imaging, tracer=tracer)

    return fit


fit = fit_with_offset_centre(centre=(2.0, 2.0))

# al.plot.fit_imaging.subplot_for_plane(fit=fit, plane_index=1, plot_source_grid=True)

al.plot.inversion.reconstruction(inversion=fit.inversion, include_grid=True)

stop

al.plot.fit_imaging.subplot(
    fit=fit,
    include_mask=True,
    points=[[(2.2, 2.2), (-0.2, -0.2), (-0.2, 2.2), (2.2, -0.2)]],
    include_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(2.05, 2.05))


al.plot.fit_imaging.subplot(
    fit=fit,
    include_mask=True,
    points=[[(2.2, 2.2), (-0.2, -0.2), (-0.2, 2.2), (2.2, -0.2)]],
    include_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(2.1, 2.1))

al.plot.fit_imaging.subplot(
    fit=fit,
    include_mask=True,
    points=[[(2.2, 2.2), (-0.2, -0.2), (-0.2, 2.2), (2.2, -0.2)]],
    include_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(2.95, 2.95))

al.plot.fit_imaging.subplot(
    fit=fit,
    include_mask=True,
    points=[[(2.2, 2.2), (-0.2, -0.2), (-0.2, 2.2), (2.2, -0.2)]],
    include_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(5.9, 5.9))

al.plot.fit_imaging.subplot(
    fit=fit,
    include_mask=True,
    points=[[(2.2, 2.2), (-0.2, -0.2), (-0.2, 2.2), (2.2, -0.2)]],
    include_image_plane_pix=True,
)
