import autolens as al
import autolens.plot as aplt

import numpy as np
import os

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../../../autolens_workspace/".format(
    os.path.dirname(os.path.realpath(__file__))
)
plot_path = "{}/../images/interferometry/".format(
    os.path.dirname(os.path.realpath(__file__))
)
dataset_path = "{}/dataset".format(os.path.dirname(os.path.realpath(__file__)))

# This loads the interferometer dataset,.
interferometer = al.Interferometer.from_fits(
    visibilities_path=dataset_path + "visibilities.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    uv_wavelengths_path=dataset_path + "uv_wavelengths.fits",
)

real_space_mask = al.Mask.circular(
    shape_2d=(151, 151), pixel_scales=0.05, sub_size=4, radius=3.0
)

masked_interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    visibilities_mask=np.full(
        fill_value=False, shape=interferometer.visibilities.shape
    ),
    real_space_mask=real_space_mask,
)

# Setup the lens galaxy's light (elliptical Sersic), mass (SIE+Shear) and source galaxy light (elliptical Sersic) for
# this simulated lens.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=60.0,
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

# Use these galaxies to setup a tracer, which will generate the image for the simulated interferometer dataset.
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Image Before Fourier Transform"),
    output=aplt.Output(path=plot_path, filename="image_pre_ft", format="png"),
)

aplt.Tracer.profile_image(
    tracer=tracer, grid=masked_interferometer.grid, plotter=plotter
)

fit = al.FitInterferometer(masked_interferometer=masked_interferometer, tracer=tracer)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Model Visibilities"),
    output=aplt.Output(path=plot_path, filename="model_visibilities", format="png"),
)

aplt.FitInterferometer.model_visibilities(fit=fit, plotter=plotter)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Residual Map vs UV-distances (Real)"),
    output=aplt.Output(path=plot_path, filename="residual_map", format="png"),
)

aplt.FitInterferometer.residual_map_vs_uv_distances(
    fit=fit, plot_real=True, plotter=plotter
)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Residual Map vs UV-distances (Imaginary)"),
    output=aplt.Output(path=plot_path, filename="residual_map", format="png"),
)

aplt.FitInterferometer.residual_map_vs_uv_distances(
    fit=fit, plot_real=False, plotter=plotter
)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Chi-Squared Map vs UV-distances (Real)"),
    output=aplt.Output(path=plot_path, filename="chi_squared_map", format="png"),
)

aplt.FitInterferometer.chi_squared_map_vs_uv_distances(
    fit=fit, plot_real=True, plotter=plotter
)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Chi-Squared Map vs UV-distances (Imaginary)"),
    output=aplt.Output(path=plot_path, filename="chi_squared_map", format="png"),
)

aplt.FitInterferometer.chi_squared_map_vs_uv_distances(
    fit=fit, plot_real=False, plotter=plotter
)
