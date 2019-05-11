import torch

from DisneyBlock import DisneyBlock


class ProbeRendererModel(torch.nn.Module):
    DESCRIPTOR_LAYER_DIMENSION = 5 * 5 * 9
    DESCRIPTOR_LAYER_WITH_META_DIMENSION = DESCRIPTOR_LAYER_DIMENSION + 5

    def __init__(self, lightProbeDimension):
        """
        :param lightProbeDimension: length of 1d vector, which will be representing the light probe
        """
        super(ProbeRendererModel, self).__init__()

        self.inputDimension = lightProbeDimension

        self.blocks = self.__createBlocks()
        self.fullyConnected = self.__createFullyConnected()

    def forward(self, lightProbe, disneyDescriptor):
        """
        :param lightProbe: light probe with an two angles and one offset vector
        :return: ouput of current block
        """

        batchSize = disneyDescriptor.size()[0]
        out = torch.zeros((batchSize, self.inputDimension))
        for i, block in enumerate(self.blocks):
            out = block(out, disneyDescriptor.narrow(1, i, 1).squeeze(1))

        return self.fullyConnected(torch.cat((out, lightProbe), 1))

    def __createBlocks(self):
        return torch.nn.ModuleList([
            self.__createBlock(),
            self.__createBlock(),
            self.__createBlock(),
            self.__createBlock(),
            self.__createBlock(),
            self.__createBlock(),
            self.__createBlock(),
            self.__createBlock(),
            self.__createBlock(),
        ])

    def __createBlock(self):
        return DisneyBlock(
            self.inputDimension,
            self.DESCRIPTOR_LAYER_WITH_META_DIMENSION,
            self.inputDimension
        )

    def __createFullyConnected(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.inputDimension * 2, self.inputDimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.inputDimension, self.inputDimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.inputDimension, 1),
            torch.nn.LeakyReLU()
        )