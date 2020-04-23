import autolens as al
import autolens.plot as aplt
import os

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../../../autolens_workspace/".format(
    os.path.dirname(os.path.realpath(__file__))
)
plot_path = "{}/../images/fitting/".format(os.path.dirname(os.path.realpath(__file__)))
dataset_path = "{}/dataset/".format(os.path.dirname(os.path.realpath(__file__)))

# Using the dataset path, load the data (image, noise map, PSF) as an imaging object from .fits files.
imaging = al.Imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=0.05,
)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=4, radius=3.0
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

# Setup the lens galaxy's light (elliptical Sersic), mass (SIE+Shear) and source galaxy light (elliptical Sersic) for
# this simulated lens.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.lp.EllipticalSersic(
        centre=(0.00, 0.00),
        axis_ratio=0.9,
        phi=45.0,
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.05, 0.05), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=60.0,
        intensity=0.6,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


# Use these galaxies to setup a tracer, which will generate the image for the simulated imaging dataset.
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Bad Residual Map"),
    output=aplt.Output(path=plot_path, filename="bad_residual_map", format="png"),
)

aplt.FitImaging.residual_map(fit=fit, plotter=plotter)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Bad Normalized Residual Map"),
    output=aplt.Output(
        path=plot_path, filename="bad_normalized_residual_map", format="png"
    ),
)

aplt.FitImaging.normalized_residual_map(fit=fit, plotter=plotter)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Bad Chi-Squared Map"),
    output=aplt.Output(path=plot_path, filename="bad_chi_squared_map", format="png"),
)

aplt.FitImaging.chi_squared_map(fit=fit, plotter=plotter)
