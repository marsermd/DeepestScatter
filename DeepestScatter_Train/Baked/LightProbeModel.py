import torch

from DisneyBlock import DisneyBlock


class LightProbeModel(torch.nn.Module):
    BLOCK_DIMENSION = 200
    LAYER_DIMENSION = 400
    DESCRIPTOR_LAYER_DIMENSION = 5 * 5 * 9

    def __init__(self, outputDimenstion, blockCount):
        """
        :param outputDimenstion: length of 1d vector, which will be representing the light probe
        """
        super(LightProbeModel, self).__init__()

        self.outputDimenstion = outputDimenstion
        self.blockCount = blockCount

        self.blocks = self.__createBlocks()
        self.fullyConnected = self.__createFullyConnected()

    def forward(self, zLayers):
        """
        :param zLayers: hierarchical descriptor as a 2D tensor, with 1D layers
        :return representation of the light for this descriptor. I.e. a lightprobe.
        """

        assert(zLayers.size()[2] == self.DESCRIPTOR_LAYER_DIMENSION)

        out = self.__blocksForward(zLayers)
        out = self.fullyConnected(out)

        return out

    def __blocksForward(self, zLayers):
        batchSize = zLayers.size()[0]

        out = torch.zeros((batchSize, self.BLOCK_DIMENSION))
        for i, block in enumerate(self.blocks):
            out = block(out, zLayers.narrow(1, self.blockCount - i - 1, 1).squeeze(1))
        return out

    def __createBlocks(self):
        return torch.nn.ModuleList(
            [self.__createBlock() for i in range(self.blockCount)]
        )

    def __createBlock(self):
        return DisneyBlock(
            self.BLOCK_DIMENSION,
            self.DESCRIPTOR_LAYER_DIMENSION,
            self.BLOCK_DIMENSION
        )

    def __createFullyConnected(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.BLOCK_DIMENSION, self.LAYER_DIMENSION),
            torch.nn.ReLU(),
            torch.nn.Linear(self.LAYER_DIMENSION, self.LAYER_DIMENSION),
            torch.nn.ReLU(),
            torch.nn.Linear(self.LAYER_DIMENSION, self.LAYER_DIMENSION),
            torch.nn.ReLU(),
            torch.nn.Linear(self.LAYER_DIMENSION, self.outputDimenstion),
            torch.nn.ReLU()
        )