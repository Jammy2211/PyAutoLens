import os

import autofit as af
import autolens as al
import autolens.plot as aplt
import numpy as np

# Setup the path to the autolens_workspace, using a relative directory name.
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

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Visibilities"),
    output=aplt.Output(path=plot_path, filename="visibilities", format="png"),
)

aplt.Interferometer.visibilities(interferometer=interferometer, plotter=plotter)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="UV-Wavelengths"),
    output=aplt.Output(path=plot_path, filename="uv_wavelengths", format="png"),
)

aplt.Interferometer.visibilities(interferometer=interferometer, plotter=plotter)
