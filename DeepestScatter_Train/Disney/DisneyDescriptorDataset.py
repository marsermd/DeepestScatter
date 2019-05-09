import torch
import numpy as np

from BaseDataset import BaseDataset
from PythonProtocols.DisneyDescriptor_pb2 import DisneyDescriptor
from Vector import angleBetween, npVector


class DisneyDescriptorDataset(BaseDataset):
    def __init__(self, lmdbDataset):
        super(DisneyDescriptorDataset, self).__init__(lmdbDataset, DisneyDescriptor)

    def __doGetItem__(self):
        descriptor = self.__getDescriptor()
        angle = self.__getViewLightAngle()
        light = self.__getLightIntensity()

        angle = torch.ones((10, 1), dtype=torch.float32) * angle

        descriptor = torch.cat((descriptor, angle), dim=1)
        return descriptor, light

    def __getDescriptor(self):
        descriptor = self.getDisneyDescriptor()

        # Grid values are stored as bytes. Let's convert them to 0-1 range
        descriptor = torch.tensor(list(descriptor.grid), dtype=torch.float32) / 256
        # Shape the grid according to the layers
        descriptor = descriptor.view((10, -1))

        return descriptor

    def __getViewLightAngle(self):
        scene = self.getScene()
        sample = self.getScatterSample()

        angle = angleBetween(npVector(scene.light_direction), npVector(sample.view_direction))
        return torch.tensor(angle, dtype=torch.float32)

    def __getLightIntensity(self):
        result = self.getResult()
        assert result.is_converged

        return result.light_intensity
