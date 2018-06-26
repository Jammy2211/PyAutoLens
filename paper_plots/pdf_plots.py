import sys
import os

import src.config.config

sys.path.append("../")

from src.analysis import model_mapper, non_linear
from src.profiles import mass_profiles as mp

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))
results_path = path + '../files/SLACS03/'

image_name = 'SLACSJ0252+0039'

mapper = model_mapper.ModelMapper(config=src.config.config.DefaultPriorConfig(config_folder_path=results_path),
                                  stellar_bulge=mp.EllipticalSersicMass, stellar_envelope=mp.EllipticalExponentialMass,
                                  dark_matter_halo=mp.SphericalNFW, shear=mp.ExternalShear)

results = non_linear.MultiNestFinished(path=results_path, obj_name=image_name, model_mapper=mapper)

print(results.total_parameters1)
print(results._most_likely)