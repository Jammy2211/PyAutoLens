from auto_lens.imaging import image
from auto_lens.profiles import light_profiles, mass_profiles
from auto_lens import galaxy
from auto_lens import ray_tracing

lens_galaxy = galaxy.Galaxy(mass_profiles=[mass_profiles.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.95,
                                                                              phi=0.0, einstein_radius=1.0)])

source_galaxy = galaxy.Galaxy(light_profiles=[light_profiles.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7,
                              phi=45.0, intensity=0.1, effective_radius=0.5, sersic_index=1.0)])

ray_trace = ray_tracing.TraceImageAndSource(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy])