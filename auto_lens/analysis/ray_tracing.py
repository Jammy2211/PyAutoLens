import numpy as np

def compute_deflection_angles(coordinates, galaxy):
    """Compute the deflection angles at a given set of sub-gridded coordinates for an input set of mass profiles

    Parameters
    -----------
    coordinates : ndarray
        The coordinates the deflection angles are computed on.
    mass_profiles : [Galaxy]

    """
    if galaxy.mass_profiles != None:
        return deflection_angles_analysis_array(galaxy, coordinates)




# TODO : I don't expect these to be the best way to do this - I'm just getting them up so I can develop the \
# TODO : ray_tracing module. I'm relying on your coding trickery to come up with a neat way of computing light and \
# TODO : mass profiles given a NumPy array.

# TODO : I'm expecting there'll be a general solution to performing the calculations below, which can go somewhere \
# TODO : in the profiles module

# TODO : If we require bespoke routines for each structure, we could make them class methods in the analysis_data \
# TODO : module, e.g. AnalysisCoordinates.compute_deflection_angles(galaxy) and AnalysisSubCoordinates.compute_defl...

def deflection_angles_analysis_array(galaxy, coordinates):
    """Compute the deflections angles for a mass profile, at a set of coordinates using the analysis_data structure.
    """
    deflection_angles = np.zeros(coordinates.shape)

    for defl_count, [y, x] in enumerate(coordinates):
        deflection_angles[defl_count, :] = galaxy.deflection_angles_at_coordinates(coordinates=(y,x))

    return deflection_angles

def deflection_angles_analysis_sub_array(galaxy, sub_coordinates):
    """Compute the deflections angles for a mass profile, at a set of sub coordinates using the analysis_data \
    structure
    """
    deflection_angles = np.zeros(sub_coordinates.shape)

    for sub_count, sub_coordinate in enumerate(sub_coordinates):
        for defl_count, [y,x] in enumerate(sub_coordinate):

            deflection_angles[sub_count, defl_count, :] = galaxy.deflection_angles_at_coordinates(coordinates=(y,x))

    return deflection_angles