import torch
from torch.utils import data

import numpy as np

from PythonProtocols.SceneSetup_pb2 import SceneSetup
from PythonProtocols.ScatterSample_pb2 import ScatterSample
from PythonProtocols.DisneyDescriptor_pb2 import DisneyDescriptor
from PythonProtocols.Result_pb2 import Result
from GlobalSettings import BATCH_SIZE


def normalized(v):
    magnitude = np.sqrt(np.dot(v, v))
    if magnitude == 0:
        magnitude = np.finfo(v.dtype).eps
    return v / magnitude


class DisneyDescriptorDataset(data.Dataset):
    def __init__(self, lmdbDataset):
        self.lmdbDataset = lmdbDataset

    def __len__(self):
        return self.lmdbDataset.getCountOf(DisneyDescriptor)

    def __getitem__(self, index):
        sampleId = index

        descriptor = self.__getDescriptor(sampleId)
        angle = self.__getLightAngle(sampleId)
        light = self.__getLightIntensity(sampleId)

        return descriptor, angle, light

    def __getDescriptor(self, sampleId):
        descriptor = self.lmdbDataset.get(DisneyDescriptor, sampleId)

        # Grid values are stored as bytes. Let's convert them to 0-1 range
        descriptor = torch.tensor(list(descriptor.grid), dtype=torch.float32) / 128

        # Shape the grid according to the layers
        descriptor = descriptor.reshape((10, 225))

        return descriptor

    def __getLightAngle(self, sampleId):
        sceneId = sampleId // BATCH_SIZE

        sample = self.lmdbDataset.get(ScatterSample, sampleId)
        scene = self.lmdbDataset.get(SceneSetup, sceneId)

        lightDirection = np.array([scene.light_direction.x, scene.light_direction.y, scene.light_direction.z])
        viewDirection = np.array([sample.view_direction.x, sample.view_direction.y, sample.view_direction.z])

        angle = np.arccos(np.dot(normalized(lightDirection), normalized(viewDirection)))
        return torch.tensor(angle, dtype=torch.float32)

    def __getLightIntensity(self, sampleId):
        result = self.lmdbDataset.get(Result, sampleId)
        assert result.is_converged

        return result.light_intensity
