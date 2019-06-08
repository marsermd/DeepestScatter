import torch

from BakedInterpolationSet_pb2 import BakedInterpolationSet
from BaseDataset import BaseDataset
from Vector import angleBetween, npVector, descriptorBasis, signedAngleBetween


class BakedDataset(BaseDataset):

    def __init__(self, lmdbDataset, bakedLayers, realtimeLayers):
        super(BakedDataset, self).__init__(lmdbDataset, BakedInterpolationSet)
        self.bakedLayers = bakedLayers
        self.realtimeLayers = realtimeLayers

    def __doGetItem__(self):
        bakedDescriptors, powers = self.__getBakedDescriptors()
        omega = self.__getViewLightAngle()
        alpha = self.__getDescriptorAngle()
        light = self.__getLightIntensity()

        disneyDescriptor = self.__getDisneyDescriptor()
        disneyDescriptor = torch.cat((disneyDescriptor, omega.repeat(self.realtimeLayers, 1)), dim=1)

        return (bakedDescriptors, powers, disneyDescriptor, omega, alpha), light

    def __getBakedDescriptors(self):
        interpolationSet = self.getBakedInterpolationSet()

        descriptors = [interpolationSet.a, interpolationSet.b, interpolationSet.c, interpolationSet.d]
        powers = [torch.tensor([descriptor.power], dtype=torch.float32) for descriptor in descriptors]

        # Grid values are stored as bytes. Let's convert them to 0-1 range
        descriptors = [torch.tensor(list(descriptor.grid), dtype=torch.float32) / 256 for descriptor in descriptors]
        # Shape the grid according to the layers
        descriptors = [descriptor.view((10, -1)).narrow(0, 0, self.bakedLayers) for descriptor in descriptors]

        return descriptors, powers

    def __getDisneyDescriptor(self):
        descriptor = self.getDisneyDescriptor()

        # Grid values are stored as bytes. Let's convert them to 0-1 range
        descriptor = torch.tensor(list(descriptor.grid), dtype=torch.float32) / 256
        # Shape the grid according to the layers
        descriptor = descriptor.view((10, -1)).narrow(0, 0, self.realtimeLayers)

        return descriptor

    def __getViewLightAngle(self):
        scene = self.getScene()
        sample = self.getScatterSample()

        angle = angleBetween(npVector(scene.light_direction), npVector(sample.view_direction))
        return torch.tensor(angle, dtype=torch.float32)

    def __getDescriptorAngle(self):
        scene = self.getScene()
        sample = self.getScatterSample()
        # We can take any of the 4 descriptors. They will all have the same angle.
        descriptor = self.getBakedInterpolationSet().a

        lightDirection = npVector(scene.light_direction)

        x1, y1, z1 = descriptorBasis(lightDirection, npVector(sample.view_direction))
        x2, y2, z2 = descriptorBasis(lightDirection, npVector(descriptor.direction))

        alpha = signedAngleBetween(y1, y2, z1)

        return torch.tensor(alpha, dtype=torch.float32)

    def __getLightIntensity(self):
        result = self.getResult()
        assert result.is_converged

        return torch.tensor(result.light_intensity)
