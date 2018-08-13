from autolens import conf


class TestCase(object):
    def test_is_config(self):
        assert conf.is_config()
