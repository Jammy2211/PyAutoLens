from autolens import config
import requests


class TestCase(object):
    def test_is_config(self):
        assert config.is_config()

    def test_is_hosted(self):
        with requests.get(config.CONFIG_URL) as response:
            assert response.status_code == 200
