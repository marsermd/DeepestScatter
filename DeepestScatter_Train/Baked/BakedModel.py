import torch
import math
import numpy as np

from LightProbeModel import LightProbeModel
from ProbeRenderModel import ProbeRendererModel


class BakedModel(torch.nn.Module):
    LIGHT_PROBE_DIMENSION = 200
    LIGHT_PROBE_DIMENSION_WITH_META = LIGHT_PROBE_DIMENSION + 2
    LIGHT_PROBE_ROTATION_STEP = 1

    def __init__(self, bakedLayers, realtimeLayers):
        super(BakedModel, self).__init__()

        self.lightProbeModel = LightProbeModel(self.LIGHT_PROBE_DIMENSION, bakedLayers)
        self.rendererModel = ProbeRendererModel(self.LIGHT_PROBE_DIMENSION_WITH_META, realtimeLayers)

    @staticmethod
    def roll(tensor, rowShifts):
        return torch.stack(list(map(torch.roll, torch.unbind(tensor, 0), rowShifts)), 0)

    def applyAnglesToLightProbe(self, lightprobe, omega, alpha):
        """
        :param lightprobe: tensor of (200), interpolated fropm the 4 closest vertices.
        :param omega: omega of the disney descriptor
        :param alpha: angle to rotate the lightprobe for, from -pi to pi
        :return: rotated lightprobe
        """

        # alpha /= math.pi
        # alpha = alpha.cpu().numpy()
        # shifts = [0] * alpha.size
        #
        # for i, a in enumerate(alpha):
        #     if a < 0:
        #         a += 2
        #
        #     factor = self.LIGHT_PROBE_DIMENSION / 2 / self.LIGHT_PROBE_ROTATION_STEP
        #     shifts[i] = int(round(a * factor))
        #     alpha[i] = a - shifts[i] / factor
        #
        #     shifts[i] *= self.LIGHT_PROBE_ROTATION_STEP
        #
        # alpha = torch.tensor(alpha)

        return torch.cat(
            (
                lightprobe,
                omega.unsqueeze(1),
                alpha.unsqueeze(1)
            ),
            dim=1
        )

    def forward(self, lightProbeDescriptors, lightProbePowers, disneyDescriptor, omega, alpha):
        """
        :param lightProbeDescriptor: hierarchical descriptor for baking the light
        :param disneyDescriptor: hierarchical descriptor for runtime rendering
        :param omega: angle between the light and view direction
        :param alpha: angle between the view oriented box and a "forward" oriented box
        :param offset: a 3d vector, representing the offset between the lightProbe and the given point.
        Given In coordinate system of the light probe box.
        :param out: ouput of current block
        """

        lightProbes = [self.lightProbeModel(descriptor) for descriptor in lightProbeDescriptors]
        lightProbe = sum([lightProbe * power.repeat(1, self.LIGHT_PROBE_DIMENSION)
                          for (lightProbe, power) in zip(lightProbes, lightProbePowers)])


        #TODO: add angles to disney's descriptor?
        #TODO: Or maybe rotate the probe??

        lightProbe = self.applyAnglesToLightProbe(lightProbe, omega, alpha)

        out = self.rendererModel(lightProbe, disneyDescriptor)

        return out