import autolens as al


def test__hyper_galaxies_names_for_lens_and_source():
    setup = al.legacy.SetupAdapt(hyper_galaxies_lens=False, hyper_galaxies_source=False)
    assert setup.hyper_galaxies is False
    assert setup.hyper_galaxy_names == None

    setup = al.legacy.SetupAdapt(hyper_galaxies_lens=True, hyper_galaxies_source=False)
    assert setup.hyper_galaxies is True
    assert setup.hyper_galaxy_names == ["lens"]

    setup = al.legacy.SetupAdapt(hyper_galaxies_lens=False, hyper_galaxies_source=True)
    assert setup.hyper_galaxies is True
    assert setup.hyper_galaxy_names == ["source"]

    setup = al.legacy.SetupAdapt(hyper_galaxies_lens=True, hyper_galaxies_source=True)
    assert setup.hyper_galaxies is True
    assert setup.hyper_galaxy_names == ["lens", "source"]
