import numpy as np
from scipy.special import logsumexp


def test_andrew_implementation():
    data = np.array([(0.0, 0.0), (1.0, 0.0)])
    model_positions = np.array([(-1.0749, -1.1), (1.19117, 1.175)])

    error = 1.0
    noise_map = np.array([error, error])

    expected = -4.40375330990644

    model_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    cov = np.identity(2) * error**2

    Ltot = 0
    Larray = []

    def square_distance(coord1, coord2):
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

    for permutation in model_indices:
        chi2 = (
            square_distance(data[0], model_positions[permutation[0]]) / error**2
            + square_distance(data[1], model_positions[permutation[1]]) / error**2
        )
        L = (
            (1 / np.sqrt(np.linalg.det(2 * np.pi * cov)))
            * np.exp(-0.5 * chi2)
            / len(model_indices)
        )
        Larray.append(L)
        Ltot += L

    print(np.log(Ltot))

    def logP(pos, model_pos, sigma=error):
        chi2 = square_distance(pos, model_pos) / sigma**2
        return -np.log(np.sqrt(2 * np.pi * sigma**2)) - 0.5 * chi2

    log_likelihood = -np.log(4) + np.sum(
        np.array(
            [
                logsumexp([logP(obs_pos, model_pos) for model_pos in model_positions])
                for obs_pos in data
            ]
        )
    )

    print(log_likelihood)

    log_likelihood = -np.log(4) + np.sum(
        np.array(
            [
                logsumexp([logP(data[i], model_positions[p]) for p in [0, 1]])
                for i in [0, 1]
            ]
        )
    )

    print(log_likelihood)

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
