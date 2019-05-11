import torch

from LightProbeModel import LightProbeModel
from ProbeRenderModel import ProbeRendererModel


class BakedModel(torch.nn.Module):
    LIGHT_PROBE_DIMENSION = 200
    LIGHT_PROBE_DIMENSION_WITH_META = LIGHT_PROBE_DIMENSION + 2

    def __init__(self, bakedLayers, realtimeLayers):
        super(BakedModel, self).__init__()

        self.lightProbe = LightProbeModel(self.LIGHT_PROBE_DIMENSION, bakedLayers)
        self.renderer = ProbeRendererModel(self.LIGHT_PROBE_DIMENSION_WITH_META, realtimeLayers)

    def forward(self, lightProbeDescriptor, disneyDescriptor, omega, alpha):
        """
        :param lightProbeDescriptor: hierarchical descriptor for baking the light
        :param disneyDescriptor: hierarchical descriptor for runtime rendering
        :param omega: angle between the light and view direction
        :param alpha: angle between the view oriented box and a "forward" oriented box
        :param offset: a 3d vector, representing the offset between the lightProbe and the given point.
        Given In coordinate system of the light probe box.
        :param out: ouput of current block
        """

        lightProbe = self.lightProbe(lightProbeDescriptor)

        lightProbe = torch.cat(
            (
                lightProbe,
                omega.unsqueeze(1),
                alpha.unsqueeze(1)
            ),
            dim=1
        )

        out = self.renderer(lightProbe, disneyDescriptor)

        return out