import getdist


def generate_parameter_latex(parameters, subscript=''):
    """Generate a latex label for a parameter, typically used for the MultiNest / getdist paramnames file and \
    outputting the results of a MultiNest run to a latex table.

    Parameters
    ----------
    parameters : [str]
        The parameter names to be converted to latex.
    subscript : str
        The subscript of the latex entry, often giving the parameter type (e.g. light or dark matter) or numerical \
        number of the component of the model.

    """

    latex = []

    if subscript == '':
        for param in parameters:
            latex.append('$' + param + '$')
    else:
        for param in parameters:
            latex.append('$' + param + r'_{\mathrm{' + subscript + '}}$')

    return latex


def read_most_probable(results_path):
    """
    Read the most probable model from the results of a MultiNest non-linear sampling run. This is performed by using \
    the 'summary.txt' file, which stores the parameters of the most probable model in its first half of entries.

    The most probable model is defined as the model which is the mean value of all samplings of a parameter weighted \
    by their sampling probabilities.

    Parameters
    ----------
    results_path : str
        A string pointing to the directory with the MultiNest results (e.g. weighted_samples.txt, phy_live.points, \
        stats.dat, summary.txt, etc.)

    """

    summary = open(results_path + 'summary.txt')

    total_parameters = compute_total_parameters(summary)

    skip = summary.read(2)  # skip the first 3 characters of the file, which are an indentation

    most_probable_vector = []

    for param in range(total_parameters):
        most_probable_vector.append(float(summary.read(28)))

    summary.close()

    return most_probable_vector


def read_most_likely(results_path):
    """
    Read the most probable model from the results of a MultiNest non-linear sampling run. This is performed by using \
    the 'summary.txt' file, which stores the parameters of the most probable model in its first half of entries.

    The most likely model is the set of parameters corresponding to the highest likelihood solution.

    Parameters
    ----------
    results_path : str
        A string pointing to the directory with the MultiNest results (e.g. weighted_samples.txt, phy_live.points, \
        stats.dat, summary.txt, etc.)

    """

    summary = open(results_path + 'summary.txt')

    total_parameters = compute_total_parameters(summary)

    skip = summary.read(2 + 28 * total_parameters)  # skip the first 3 characters of the file, which are an indentation

    most_likely_vector = []

    for param in range(total_parameters):
        most_likely_vector.append(float(summary.read(28)))

    summary.close()

    return most_likely_vector


def compute_total_parameters(summary, reset_position=True):
    """ Each parameter in the summary file is 28 characters long (including its 4 spaces). Parameters are listed twice \
    (most probable and most likely models) and there are two extra 28 character slots for the most likley model's \
    likelihood and loglikelihood.

    Therefore, the total parameters be computed from the summary file by diving the length of its top line by 28, \
    halving this value and subtracting 2."""

    total_parameters = ((len(summary.readline()) // 28) // 2) - 1
    if reset_position == True:
        summary.seek(0)

    return total_parameters
