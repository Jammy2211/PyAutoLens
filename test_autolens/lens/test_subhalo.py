import autofit as af
import autolens as al

def test__detection_array_from():

    samples_list = [
        [[af.mock.MockSamples(log_likelihood_list=[1.0]),
        af.mock.MockSamples(log_likelihood_list=[2.0])],
        [af.mock.MockSamples(log_likelihood_list=[3.0]),
        af.mock.MockSamples(log_likelihood_list=[4.0])]],
    ]

    grid_search_result_with_subhalo = af.GridSearchResult(
        lower_limits_lists=[[1.0, 2.0], [3.0, 4.0]],
        samples=samples_list,
        grid_priors=[[1, 2], [3, 4]]
    )

    subhalo_result = al.subhalo.SubhaloResult(
        grid_search_result_with_subhalo=grid_search_result_with_subhalo,
        fit_imaging_no_subhalo=None,
        samples_no_subhalo=None
    )

    detection_array = subhalo_result.detection_array_from(
        use_log_evidences=False,
        relative_to_no_subhalo=False,
        remove_zeros=False
    )

    print(detection_array)