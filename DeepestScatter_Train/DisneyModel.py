import torch

from DisneyBlock import DisneyBlock

class DisneyModel(torch.nn.Module):
    BLOCK_DIMENSION = 200
    BLOCK_COUNT = 10
    DESCRIPTOR_LAYER_DIMENSION = 5 * 5 * 9
    DESCRIPTOR_LAYER_WITH_ANGLE_DIMENSION = DESCRIPTOR_LAYER_DIMENSION + 1

    FULLY_CONNECTED_COUNT = 3

    def __init__(self):
        super(DisneyModel, self).__init__()
        self.blocks = self.__createBlocks()
        self.fullyConnected = self.__createFullyConnected()

    def forward(self, zLayers):
        """
        :param zLayers: hierarchical descriptor as a 2D tensor, with 1D layers,
         with an angle between light and view direction appended to each layer.
        :return estimated radiance, given that light has radiance of 1e6
        """

        assert(zLayers.size()[1] == self.BLOCK_COUNT)
        assert(zLayers.size()[2] == self.DESCRIPTOR_LAYER_WITH_ANGLE_DIMENSION)

        out = self.__blocksForward(zLayers)
        out = self.fullyConnected(out)

        return out

    def __blocksForward(self, zLayers):
        batchSize = zLayers.size()[0]

        out = torch.zeros((batchSize, self.BLOCK_DIMENSION))
        for i, block in enumerate(self.blocks):
            out = block(out, zLayers.narrow(1, i, 1).squeeze(1))
        return out

    def __createBlocks(self):
        return torch.nn.ModuleList(
            [self.__createBlock() for i in range(self.BLOCK_COUNT)]
        )

    def __createBlock(self):
        return DisneyBlock(
            self.BLOCK_DIMENSION,
            self.DESCRIPTOR_LAYER_WITH_ANGLE_DIMENSION,
            self.BLOCK_DIMENSION
        )

    def __createFullyConnected(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.BLOCK_DIMENSION, self.BLOCK_DIMENSION),
            torch.nn.ReLU(),
            torch.nn.Linear(self.BLOCK_DIMENSION, self.BLOCK_DIMENSION),
            torch.nn.ReLU(),
            torch.nn.Linear(self.BLOCK_DIMENSION, 1)
        )