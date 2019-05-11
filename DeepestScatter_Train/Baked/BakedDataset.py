import torch
import numpy as np

from BaseDataset import BaseDataset
from PythonProtocols.BakedDescriptor_pb2 import BakedDescriptor
from Vector import angleBetween, npVector, descriptorBasis, projectionOn


class BakedDataset(BaseDataset):

    def __init__(self, lmdbDataset, bakedLayers, realtimeLayers):
        self.bakedLayers = bakedLayers
        self.realtimeLayers = realtimeLayers
        super(BakedDataset, self).__init__(lmdbDataset, BakedDescriptor)

    def __doGetItem__(self):
        bakedDescriptor = self.__getBakedDescriptor()
        omega = self.__getViewLightAngle()
        alpha = self.__getDescriptorAngle()
        light = self.__getLightIntensity()

        disneyDescriptor = torch.cat(
            (
                self.__getDisneyDescriptor(),
                omega.repeat(self.realtimeLayers, 1),
                alpha.repeat(self.realtimeLayers, 1)
            ), dim=1)

        return (bakedDescriptor, disneyDescriptor, omega, alpha), light

    def __getBakedDescriptor(self):
        descriptor = self.getBakedDescriptor()

        # Grid values are stored as bytes. Let's convert them to 0-1 range
        descriptor = torch.tensor(list(descriptor.grid), dtype=torch.float32) / 256
        # Shape the grid according to the layers
        descriptor = descriptor.view((10, -1)).narrow(0, 0, self.bakedLayers)

        return descriptor

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
        descriptor = self.getBakedDescriptor()

        lightDirection = npVector(scene.light_direction)

        x1, y1, z1 = descriptorBasis(lightDirection, npVector(sample.view_direction))
        x2, y2, z2 = descriptorBasis(lightDirection, npVector(descriptor.direction))

        alpha = angleBetween(y1, y2)

        return torch.tensor(alpha, dtype=torch.float32)


    def __getLightIntensity(self):
        result = self.getResult()
        assert result.is_converged

        return result.light_intensity
