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

    def forward(self, descriptor, angle):
        """
        :param descriptor: hierarchical descriptor as an array of layers
        :param angle: angle between light and view direction
        :return estimated radiance given that light has radiance of 1e6
        """
        assert(descriptor.size()[1:] == (self.BLOCK_COUNT, self.DESCRIPTOR_LAYER_DIMENSION))

        batchSize = descriptor.size()[0]
        out = self.__blocksForward(batchSize, angle, descriptor)
        out = self.fullyConnected(out)

        return out

    def __blocksForward(self, batchSize, angle, descriptor):
        device = next(self.parameters()).device
        out = descriptor.new_zeros((batchSize, self.BLOCK_DIMENSION))
        for i, block in enumerate(self.blocks):
            descriptorLayer = descriptor[:, i, :]
            z = torch.cat((descriptorLayer, angle.unsqueeze(-1)), dim=-1)
            out = block(out, z)
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
            torch.nn.Linear(self.BLOCK_DIMENSION, self.BLOCK_DIMENSION),
            torch.nn.Linear(self.BLOCK_DIMENSION, 1)
        )