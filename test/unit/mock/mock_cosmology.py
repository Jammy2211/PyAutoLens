class Value(object):

    def __init__(self, value):

        self.value = value

class MockCosmology(object):

    def __init__(self, arcsec_per_kpc=0.5, kpc_per_arcsec=2.0):

        self.arcsec_per_kpc = arcsec_per_kpc
        self.kpc_per_arcsec = kpc_per_arcsec

    def arcsec_per_kpc_proper(self, z):
        return Value(value=self.arcsec_per_kpc)

    def kpc_per_arcsec_proper(self, z):
        return Value(value=self.kpc_per_arcsec)