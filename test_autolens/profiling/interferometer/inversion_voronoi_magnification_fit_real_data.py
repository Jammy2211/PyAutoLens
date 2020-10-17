import time

import autolens as al
import autolens.plot as aplt
import numpy as np

repeats = 1

workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace"
dataset_type = "interferometer"
dataset_name = "mass_sie__source_bulge"
dataset_path = f"dataset/{dataset_type}/{dataset_name}"

interferometer = al.Interferometer.from_fits(
    visibilities_path=f"{dataset_path}/visibilities.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    uv_wavelengths_path=f"{dataset_path}/uv_wavelengths.fits",
)

# lens_galaxy = al.Galaxy(
#     redshift=0.5,
#     mass=al.mp.EllipticalIsothermal(
#         centre=(0.4, 0.8), einstein_radius=0.1, elliptical_comps=(0.17647, 0.0)
#     ),
# )

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, phi=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.0, 0.05)),
)

pixelization = al.pix.VoronoiMagnification(shape=(30, 30))

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=10.0),
)

mask = al.Mask2D.circular(
    shape_2d=(256, 256), pixel_scales=0.05, sub_size=1, radius=3.0
)

masked_interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    real_space_mask=mask,
    visibilities_mask=np.full(
        fill_value=False, shape=interferometer.visibilities.shape
    ),
    transformer_class=al.TransformerNUFFT,
)

print("Number of points = " + str(masked_interferometer.grid.sub_shape_1d) + "\n")
print(
    "Number of visibilities = "
    + str(masked_interferometer.visibilities.shape_1d)
    + "\n"
)

start_overall = time.time()

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

start = time.time()

for i in range(repeats):
    fit = al.FitInterferometer(
        masked_interferometer=masked_interferometer,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_linear_operators=True),
    )

print(fit.log_evidence)

diff = time.time() - start
print("Time to compute fit = {}".format(diff / repeats))

aplt.FitInterferometer.subplot_fit_real_space(fit=fit)
aplt.Inversion.reconstruction(inversion=fit.inversion)

start = time.time()

for i in range(repeats):
    inversion = tracer.inversion_interferometer_from_grid_and_data(
        grid=masked_interferometer.grid,
        visibilities=masked_interferometer.visibilities,
        noise_map=masked_interferometer.noise_map,
        transformer=masked_interferometer.transformer,
        settings_inversion=al.SettingsInversion(use_linear_operators=True),
    )

diff = time.time() - start
print("Time to compute inversion = {}".format(diff / repeats))

aplt.FitInterferometer.subplot_fit_real_space(fit=fit)
aplt.Inversion.reconstruction(inversion=fit.inversion)
