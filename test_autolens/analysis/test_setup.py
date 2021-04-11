import autolens as al


class TestSetupHyper:
    def test__hyper_galaxies_names_for_lens_and_source(self):

        setup = al.SetupHyper(hyper_galaxies_lens=False, hyper_galaxies_source=False)
        assert setup.hyper_galaxies is False
        assert setup.hyper_galaxy_names == None

        setup = al.SetupHyper(hyper_galaxies_lens=True, hyper_galaxies_source=False)
        assert setup.hyper_galaxies is True
        assert setup.hyper_galaxy_names == ["lens"]

        setup = al.SetupHyper(hyper_galaxies_lens=False, hyper_galaxies_source=True)
        assert setup.hyper_galaxies is True
        assert setup.hyper_galaxy_names == ["source"]

        setup = al.SetupHyper(hyper_galaxies_lens=True, hyper_galaxies_source=True)
        assert setup.hyper_galaxies is True
        assert setup.hyper_galaxy_names == ["lens", "source"]
