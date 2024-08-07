import autolens as al


def test__info():
    dataset = al.PointDataset(
        "name",
        positions=al.Grid2DIrregular([(1, 2)]),
        positions_noise_map=al.ArrayIrregular([1]),
        fluxes=al.ArrayIrregular([2]),
        fluxes_noise_map=al.ArrayIrregular([3]),
    )

    assert "name" in dataset.info
    assert "positions : Grid2DIrregular" in dataset.info
