from auto_lens.profile.mass_profile import *

power_law = CoredSphericalPowerLawMassProfile(einstein_radius=1.0, slope=1.5, core_radius=0.5, centre=(0.0,0.0))
power_law2 = CoredSphericalPowerLawMassProfile(einstein_radius=1.0, slope=1.5, core_radius=0.5, centre=(0.0,0.0))

kappa = power_law.surface_density_at_coordinates(coordinates=(2.0, 3.0))
kappa2 = power_law2.surface_density_at_coordinates(coordinates=(2.0, 30.0))

potential = power_law.potential_at_coordinates(coordinates=(2.0, 3.0))
potential2 = power_law2.potential_at_coordinates(coordinates=(2.0, 30.0))

defls = power_law.deflection_angles_at_coordinates(coordinates=(2.0, 3.0))
defls2 = power_law2.deflection_angles_at_coordinates(coordinates=(2.0, 30.0))

print(kappa)
print(kappa2)

print(potential)
print(potential2)

print(defls)
print(defls2)