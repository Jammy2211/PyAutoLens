import sys
import os

import conf

sys.path.append("../")

from autolens.autopipe import model_mapper
from autolens.autopipe import non_linear
from autolens.profiles import mass_profiles as mp

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))
results_path = path + '../nlo/SLACS03/'

image_name = 'SLACSJ0252+0039'

mapper = model_mapper.ModelMapper(config=conf.DefaultPriorConfig(config_folder_path=results_path),
                                  stellar_bulge=mp.EllipticalSersicMass, stellar_envelope=mp.EllipticalExponentialMass,
                                  dark_matter_halo=mp.SphericalNFW, shear=mp.ExternalShear)

results = non_linear.MultiNestFinished(path=results_path, obj_name=image_name, model_mapper=mapper)

print(results.total_parameters1)
print(results._most_likely)