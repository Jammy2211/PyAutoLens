import autolens as al
import autolens.plot as aplt
import os

plot_path = "{}/../images/lensing/".format(os.path.dirname(os.path.realpath(__file__)))

grid = al.Grid.uniform(shape_2d=(50, 50), pixel_scales=0.05)

isothermal_mass_profile = al.mp.EllipticalIsothermal(
    centre=(0.0, 0.0), axis_ratio=0.8, phi=120.0, einstein_radius=1.6
)

convergence = isothermal_mass_profile.convergence_from_grid(grid=grid)
potential = isothermal_mass_profile.potential_from_grid(grid=grid)
deflections = isothermal_mass_profile.deflections_from_grid(grid=grid)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Image of Elliptical Isothermal Mass Profile Convergence"),
    output=aplt.Output(
        path=plot_path, filename="isothermal_mass_profile_convergence", format="png"
    ),
)

aplt.MassProfile.convergence(
    mass_profile=isothermal_mass_profile, grid=grid, plotter=plotter
)

plotter = aplt.Plotter(
    labels=aplt.Labels(title="Image of Elliptical Isothermal Mass Profile Potential"),
    output=aplt.Output(
        path=plot_path, filename="isothermal_mass_profile_potential", format="png"
    ),
)

aplt.MassProfile.potential(
    mass_profile=isothermal_mass_profile, grid=grid, plotter=plotter
)

plotter = aplt.Plotter(
    labels=aplt.Labels(
        title="Image of Elliptical Isothermal Mass Profile Deflections (y)"
    ),
    output=aplt.Output(
        path=plot_path, filename="isothermal_mass_profile_deflections_y", format="png"
    ),
)

aplt.MassProfile.deflections_y(
    mass_profile=isothermal_mass_profile, grid=grid, plotter=plotter
)

plotter = aplt.Plotter(
    labels=aplt.Labels(
        title="Image of Elliptical Isothermal Mass Profile Deflections (x)"
    ),
    output=aplt.Output(
        path=plot_path, filename="isothermal_mass_profile_deflections_x", format="png"
    ),
)

aplt.MassProfile.deflections_x(
    mass_profile=isothermal_mass_profile, grid=grid, plotter=plotter
)
