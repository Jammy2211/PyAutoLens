import sys
import os
sys.path.append("../")

from auto_lens import non_linear
from auto_lens import model_mapper
from auto_lens.profiles import light_profiles as lp
from auto_lens.profiles import mass_profiles as mp

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))
results_path = path + '../files/SLACS03/'

image_name = 'SLACSJ0252+0039'

mapper = model_mapper.ModelMapper(config=model_mapper.Config(config_folder_path=results_path),
                                  stellar_bulge=mp.EllipticalSersicMass, stellar_envelope=mp.EllipticalExponentialMass,
                                  dark_matter_halo=mp.SphericalNFW, shear=mp.ExternalShear)

results = non_linear.MultiNestFinished(path=results_path, obj_name=image_name, model_mapper=mapper)

print(results.total_parameters1)
print(results._most_likely)