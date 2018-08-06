from autolens import config


class TestCase(object):
    def test_is_config(self):
        assert config.is_config()
