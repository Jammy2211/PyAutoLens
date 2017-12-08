import cProfile
import decorator
import profile


def test_symmetric_profile():
    for _ in range(5):
        circular = profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                              effective_radius=0.6, sersic_index=4.0)

        circular.centre = (50, 50)
        array = decorator.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                       pixel_scale=1.0)

        assert array[50][50] > array[50][51]
        assert array[50][50] > array[49][50]
        assert array[49][50] == array[50][51]
        assert array[50][51] == array[50][49]
        assert array[50][49] == array[51][50]

        array = decorator.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                       pixel_scale=0.5)

        assert array[100][100] > array[100][101]
        assert array[100][100] > array[99][100]
        assert array[99][100] == array[100][101]
        assert array[100][101] == array[100][99]
        assert array[100][99] == array[101][100]


cProfile.run('test_symmetric_profile()')
