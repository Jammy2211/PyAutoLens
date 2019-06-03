from autolens.data import convolution
from autolens.model.inversion import convolution as inversion_convolution

class MockConvolverImage(convolution.ConvolverImage):

    def __init__(self, mask, blurring_mask, psf):

        super(MockConvolverImage, self).__init__(mask=mask, blurring_mask=blurring_mask, psf=psf)


class MockConvolverMappingMatrix(inversion_convolution.ConvolverMappingMatrix):

    def __init__(self, mask, psf):

        super(MockConvolverMappingMatrix, self).__init__(mask=mask, psf=psf)