phase1 = Phase(image, mask, DownhillSimplex())

phase1.variables.lens_galaxies = [GalaxyPrior()]
phase1.constance.source_galaxies = [Galaxy()]

result = phase1.run()
