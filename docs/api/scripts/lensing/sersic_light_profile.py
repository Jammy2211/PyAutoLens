import os

import autolens as al
import autolens.plot as aplt

plot_path = "{}/../images/lensing/".format(os.path.dirname(os.path.realpath(__file__)))

grid = al.Grid.uniform(shape_2d=(50, 50), pixel_scales=0.05)

sersic_light_profile = al.lp.EllipticalSersic(
    centre=(0.0, 0.0),
    axis_ratio=0.9,
    phi=60.0,
    intensity=0.05,
    effective_radius=2.0,
    sersic_index=4.0,
)

image = sersic_light_profile.profile_image_from_grid(grid=grid)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Image of Elliptical Sersic Light Profile"),
    output=aplt.Output(path=plot_path, filename="sersic_light_profile", format="png"),
)

aplt.LightProfile.profile_image(
    light_profile=sersic_light_profile, grid=grid, plotter=plotter
)
