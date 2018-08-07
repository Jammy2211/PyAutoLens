from autolens import conf
import requests


class TestCase(object):
    def test_is_config(self):
        assert conf.is_config()

    def test_is_hosted(self):
        with requests.get(conf.CONFIG_URL) as response:
            assert response.status_code == 200
