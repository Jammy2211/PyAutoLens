from auto_lens.profile.mass_profile import *
#
# power_law = CoredSphericalPowerLawMassProfile(einstein_radius=1.0, slope=1.4, core_radius=0.5, centre=(0.0,0.0))
# power_law2 = CoredSphericalPowerLawMassProfile(einstein_radius=1.0, slope=1.4, core_radius=0.5, centre=(0.0,0.0))
#
# kappa = power_law.surface_density_at_coordinates(coordinates=(2.0, 30.0))
# kappa2 = power_law2.surface_density_at_coordinates(coordinates=(2.0, 30.0))
#
# potential = power_law.potential_at_coordinates(coordinates=(2.0, 30.0))
# potential2 = power_law2.potential_at_coordinates(coordinates=(2.0, 30.0))
#
# defls = power_law.deflection_angles_at_coordinates(coordinates=(2.0, 30.0))
# defls2 = power_law2.deflection_angles_at_coordinates(coordinates=(2.0, 30.0))
#
# print(kappa)
# print(kappa2)
#
# print(potential)
# print(potential2)
#
# print(defls)
# print(defls2)

nfw = EllipticalNFWMassProfile(kappa_s=1.0, axis_ratio=1.0, phi=0.0, scale_radius=1.0, centre=(0.0,0.0))
nfw2 = SphericalNFWMassProfile(kappa_s=1.0, scale_radius=1.0, centre=(0.0,0.0))

kappa = nfw.surface_density_at_coordinates(coordinates=(2.0, 3.0))
kappa2 = nfw.surface_density_at_coordinates(coordinates=(4.0, 5.0))

potential = nfw.potential_at_coordinates(coordinates=(2.0, 3.0))
potential2 = nfw.potential_at_coordinates(coordinates=(4.0, 5.0))

defls = nfw.deflection_angles_at_coordinates(coordinates=(2.0, 3.0))
defls2 = nfw.deflection_angles_at_coordinates(coordinates=(4.0, 5.0))

print(kappa, kappa2, kappa/kappa2)
print(potential, potential2, potential/potential2)
print(defls)
print(defls2)
print(defls[0] / defls2[0])
print(defls[1] / defls2[1])