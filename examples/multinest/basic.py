import json
import numpy
from numpy import log, exp, pi
import scipy.stats, scipy
import pymultinest
import matplotlib.pyplot as plt

# we define the problem: we need a prior function which maps from [0:1] to the parameter space

# we only have one parameter, the position of the gaussian (ndim == 1)
# map it from the unity interval 0:1 to our problem space 0:2 under a uniform prior
def prior(cube, ndim, nparams):
	cube[0] = cube[0] * 2

# our likelihood function consists of 6 gaussians modes (solutions) at the positions
positions = numpy.array([0.1, 0.2, 0.5, 0.55, 0.9, 1.1])
width = 0.01

def loglike(cube, ndim, nparams):
	# get the current parameter (is between 0:2 now)
	pos = cube[0]
	likelihood = exp(-0.5 * ((pos - positions) / width)**2) / (2*pi*width**2)**0.5
	return log(likelihood.mean())

# number of dimensions our problem has
parameters = ["position"]
n_params = len(parameters)

# run MultiNest
pymultinest.run(loglike, prior, n_params, outputfiles_basename='out/',
	resume = False, verbose = True)
json.dump(parameters, open('out/params.json', 'w')) # save parameter names

# now run the script and analyse the output using multinest_marginals.py::
# 
#    $ python 1_1d_multimodal.py &&  multinest_marginals.py 1_1d_multimodal_out
#
# then open the file 1_1d_multimodal_outmarg.pdf
#
# Btw, ln(ev) should be ln(1 / 2)