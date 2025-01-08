import numpy as np
from scipy.special import logsumexp


def test_andrew_implementation():
    data = np.array([(0.0, 0.0), (1.0, 0.0)])
    model_positions = np.array([(-1.0749, -1.1), (1.19117, 1.175)])

    error = 1.0

    expected = -4.40375330990644

    def square_distance(coord1, coord2):
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

    def logP(pos, model_pos, sigma=error):
        chi2 = square_distance(pos, model_pos) / sigma**2
        return -np.log(np.sqrt(2 * np.pi * sigma**2)) - 0.5 * chi2

    P = len(model_positions)
    I = len(data)
    Nsigma = P**I  # no. of permutations
    log_likelihood = -np.log(Nsigma) + np.sum(
        np.array(
            [
                logsumexp([logP(data[i], model_positions[p]) for p in range(P)])
                for i in range(I)
            ]
        )
    )
    print(log_likelihood)
