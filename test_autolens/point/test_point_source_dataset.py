import autolens as al


def test__info():
    dataset = al.PointDataset(
        "name",
        positions=al.Grid2DIrregular([(1, 2)]),
        positions_noise_map=al.ArrayIrregular([1]),
        fluxes=al.ArrayIrregular([2]),
        fluxes_noise_map=al.ArrayIrregular([3]),
        time_delays=al.ArrayIrregular([4]),
        time_delays_noise_map=al.ArrayIrregular([5]),
    )

    assert "name" in dataset.info
    assert "positions : Grid2DIrregular" in dataset.info
    assert "positions_noise_map : ArrayIrregular" in dataset.info
    assert "fluxes : ArrayIrregular" in dataset.info
    assert "fluxes_noise_map : ArrayIrregular" in dataset.info
    assert "time_delays : ArrayIrregular" in dataset.info
    assert "time_delays_noise_map : ArrayIrregular" in dataset.info