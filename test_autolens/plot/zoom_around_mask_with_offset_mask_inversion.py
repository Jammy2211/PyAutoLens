import autolens as al

from test_autolens.simulators.imaging import instrument_util

imaging = instrument_util.load_test_imaging(
    dataset_name="mass_sie__source_sersic__offset_centre", instrument="vro"
)


def fit_with_offset_centre(centre):

    mask = al.Mask2D.elliptical(
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
            centre=(2.0, 2.0), einstein_radius=1.2, elliptical_comps=(0.17647, 0.0)
        ),
    )
    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.VoronoiMagnification(shape=(20, 20)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
    fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

    return fit


fit = fit_with_offset_centre(centre=(2.0, 2.0))

# aplt.FitImaging.subplot_for_plane(fit=fit, plane_index=1, plot_source_grid=True)

aplt.AbstractInversion.reconstruction(inversion=fit.inversion, include_grid=True)

stop

aplt.FitImaging.subplot_fit_imaging(
    fit=fit,
    mask=True,
    positions=[[(2.2, 2.2), (-0.2, -0.2), (-0.2, 2.2), (2.2, -0.2)]],
    include_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(2.05, 2.05))


aplt.FitImaging.subplot_fit_imaging(
    fit=fit,
    mask=True,
    positions=[[(2.2, 2.2), (-0.2, -0.2), (-0.2, 2.2), (2.2, -0.2)]],
    include_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(2.1, 2.1))

aplt.FitImaging.subplot_fit_imaging(
    fit=fit,
    mask=True,
    positions=[[(2.2, 2.2), (-0.2, -0.2), (-0.2, 2.2), (2.2, -0.2)]],
    include_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(2.95, 2.95))

aplt.FitImaging.subplot_fit_imaging(
    fit=fit,
    mask=True,
    positions=[[(2.2, 2.2), (-0.2, -0.2), (-0.2, 2.2), (2.2, -0.2)]],
    include_image_plane_pix=True,
)

fit = fit_with_offset_centre(centre=(5.9, 5.9))

aplt.FitImaging.subplot_fit_imaging(
    fit=fit,
    mask=True,
    positions=[[(2.2, 2.2), (-0.2, -0.2), (-0.2, 2.2), (2.2, -0.2)]],
    include_image_plane_pix=True,
)
